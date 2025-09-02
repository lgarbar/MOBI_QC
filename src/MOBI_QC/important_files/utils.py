import pandas as pd
import pyxdf
import tarfile
from io import BytesIO
import os
import platform
import re
import json
import numpy as np
import sounddevice as sd
from glob import glob
from tqdm import tqdm
import datetime
from stim_correction import trigger_recovery
from scipy.interpolate import interp1d


def load_default_event(xdf_filename, config_path="default_event.json"):
    with open(config_path, "r") as f:
        config = json.load(f)
    for task, event in config.items():
        if task.lower() in xdf_filename.lower():
            return event
    return None

def get_collection_date(xdf_filename:str):
    if platform.system() == 'Windows':
            return datetime.datetime.fromtimestamp(os.path.getctime(xdf_filename))
    else:
        # On Unix, getctime returns the last metadata change — not creation time
        stat = os.stat(xdf_filename)
        try:
            return datetime.datetime.fromtimestamp(stat.st_birthtime).strftime('%Y-%m-%d %H:%M:%S')
        except AttributeError:
            # Fallback: use modification time instead
            return datetime.datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
        
def get_stream_names(xdf_obj):
    if isinstance(xdf_obj, tuple):
        streams = xdf_obj[0]
    elif isinstance(xdf_obj, list):
        streams = xdf_obj
    else:
        raise TypeError(f"Unsupported type: {type(xdf_obj)}")
    return {stream['info']['name'][0]: i for i, stream in enumerate(streams)}

def import_webcam_data(xdf_filename:str):    
    cam_data, _ = pyxdf.load_xdf(xdf_filename, select_streams=[{'name': 'WebcamStream'}], verbose=False)
    frame_nums = [int(i[0]) for i in cam_data[0]['time_series']]
    time_pre = [float(i[1]) for i in cam_data[0]['time_series']]
    time_evnt_ms = [float(i[2]) for i in cam_data[0]['time_series']]
    time_post = [float(i[3]) for i in cam_data[0]['time_series']]

    cam_df = pd.DataFrame({'frame_num': frame_nums, 
                        'time_pre': time_pre, 
                        'cap_time_ms': time_evnt_ms,
                        'time_post': time_post,
                        'lsl_time_stamp': cam_data[0]['time_stamps']})

    cam_df['frame_time_sec'] = (cam_df.cap_time_ms - cam_df.cap_time_ms[0])/1000
    cam_df['lsl_time_sec'] = (cam_df.lsl_time_stamp - cam_df.lsl_time_stamp[0]) *1000
    return cam_df


def import_physio_data(xdf_filename:str):
    data, _ = pyxdf.load_xdf(xdf_filename, select_streams=[{'name': 'OpenSignals'}], verbose = False)
    column_labels = [data[0]['info']['desc'][0]['channels'][0]['channel'][i]['label'][0] for i in range(len(data[0]['info']['desc'][0]['channels'][0]['channel']))]
    df = pd.DataFrame(data[0]['time_series'], columns=column_labels)
    df['lsl_time_stamp'] = data[0]['time_stamps']
    df['time'] = df.lsl_time_stamp - df.lsl_time_stamp[0]
    return df

def import_ecg_data(xdf_filename:str):
    df = import_physio_data(xdf_filename)
    ecg_col = [col for col in df.columns if 'ECG' in col]
    if ecg_col:
        ecg_col = ecg_col[0]
        df = df[[ecg_col, 'lsl_time_stamp', 'time']]
    else:
        print('Missing ECG data in physio dataset')
        return None, None

    return ecg_col, df

def import_mic_data(xdf_filename:str):
    data, _ = pyxdf.load_xdf(xdf_filename, select_streams=[{'type': 'AudioCapture'}], verbose = False)
    df = pd.DataFrame(data[0]['time_series'], columns=['int_array'])
    df['bytestring'] = df['int_array'].apply(lambda x: np.array(x).tobytes())
    df['duration'] = (data[0]['time_stamps'] - data[0]['time_stamps'][0])/data[0]['info']['effective_srate']
    df['lsl_time_stamp'] = data[0]['time_stamps']
    df['time'] = df.lsl_time_stamp - df.lsl_time_stamp[0]
    return df

def import_video_data(xdf_filename:str):
    data, _ = pyxdf.load_xdf(xdf_filename, select_streams=[{'type': 'video'}], verbose = False)
    frame_nums = [int(i[0]) for i in data[0]['time_series']]
    time_pre = [float(i[1]) for i in data[0]['time_series']]
    time_evnt_ms = [float(i[2]) for i in data[0]['time_series']]
    time_post = [float(i[3]) for i in data[0]['time_series']]
    df = pd.DataFrame({'frame_num': frame_nums, 
                        'time_pre': time_pre, 
                        'cap_time_ms': time_evnt_ms,
                        'time_post': time_post,
                        'lsl_time_stamp': data[0]['time_stamps']})

    df['frame_time_sec'] = (df.cap_time_ms - df.cap_time_ms[0])/1000
    df['time'] = df.lsl_time_stamp - df.lsl_time_stamp[0]
    return df

def find_et_stream_name(streams:dict):
    for stream in streams:
        if any(string for string in ['eye', 'tracker'] if string in stream.lower()):
            return stream
def convert_cols_to_num(df:pd.DataFrame):
    slash_cols = [ind for ind, data in enumerate(df.loc[0]) if isinstance(data, str) and '/' in data]
    for col in df:
        if col in df.columns[slash_cols]:
            df[col] = df[col].apply(lambda data: [float(x) for x in data.split('/')])
        else:
            df[col] = pd.to_numeric(df[col])

    return df

def align_and_resample(df1, df2, srate=30, time_col="time"):
    # Ensure numeric time
    df1 = df1.copy()
    df2 = df2.copy()
    df1[time_col] = pd.to_numeric(df1[time_col], errors="coerce")
    df2[time_col] = pd.to_numeric(df2[time_col], errors="coerce")

    # Drop NaN times and sort
    df1 = df1.dropna(subset=[time_col]).sort_values(time_col)
    df2 = df2.dropna(subset=[time_col]).sort_values(time_col)

    # Find overlapping time range
    t0 = max(df1[time_col].iloc[0], df2[time_col].iloc[0])
    t1 = min(df1[time_col].iloc[-1], df2[time_col].iloc[-1])

    # 30 Hz target time vector
    target_time = np.arange(t0, t1, 1/srate)

    def resample_numeric(df):
        out = pd.DataFrame({time_col: target_time})
        for col in df.columns:
            if col == time_col:
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                f = interp1d(df[time_col], df[col], kind='linear', bounds_error=False, fill_value=np.nan)
                out[col] = f(target_time)
            else:
                # Nearest neighbor for non-numeric
                idx = np.searchsorted(df[time_col], target_time)
                idx = np.clip(idx, 0, len(df)-1)
                out[col] = df[col].values[idx]
        return out

    et_resampled = resample_numeric(df1)
    bh_resampled = resample_numeric(df2)

    # Merge on the time column
    merged = pd.merge(et_resampled, bh_resampled, on=time_col, suffixes=("_et", "_bh"))
    return merged

def import_et_data(xdf_filename:str):
    data, _ = pyxdf.load_xdf(xdf_filename, verbose = False)
    streams = get_stream_names(data)
    et_stream_name = find_et_stream_name(streams)
    data = data[streams[et_stream_name]]
    column_labels = [data['info']['desc'][0]['channels'][0]['channel'][i]['label'][0] for i in range(len(data['info']['desc'][0]['channels'][0]['channel']))]
    df = pd.DataFrame(data['time_series'], columns=column_labels)
    df = convert_cols_to_num(df)
    df['lsl_time_stamp'] = data['time_stamps']
    df['time'] = df.lsl_time_stamp - df.lsl_time_stamp[0]
    df['diff'] = df.lsl_time_stamp.diff()
    return df

def import_eeg_data(xdf_filename:str):
    data, _ = pyxdf.load_xdf(xdf_filename, select_streams=[{'type': 'EEG'}], verbose = False)
    ch_names = [f"E{i+1}" for i in range(data[0]['time_series'].shape[1])]
    df = pd.DataFrame(data[0]['time_series'], columns=ch_names) # index=data[0]['time_stamps']
    df['lsl_time_stamp'] = data[0]['time_stamps']
    #df['time'] = df.lsl_time_stamp - df.lsl_time_stamp[0]
    return df

def reformat_events(stim_df: pd.DataFrame, task:str=None):
    onset_mask = stim_df['event'].str.contains('onset', case=False, na=False)
    stim_df.loc[onset_mask, 'event'] = stim_df.loc[onset_mask, 'event'].apply(
        lambda x: re.sub(r'^[Oo]nset[\s_]+', 'Onset_', str(x))
    )

    onset_events = stim_df.loc[onset_mask, 'event'].str.replace(r'^Onset_', '', regex=True)

    for event in onset_events.unique():
        expected_offset = f"Offset_{event}"

        if not stim_df['event'].eq(expected_offset).any():
            pattern = rf'^[Oo]ffset[\s_]*{re.escape(event)}|TaskEnded[\s_]*{re.escape(event)}'
            candidate_mask = stim_df['event'].str.contains(pattern, case=False, na=False)
            stim_df.loc[candidate_mask, 'event'] = expected_offset

    if task.lower() == 'nasa':
        Onset_LeanEnd_ind = np.where(stim_df['event']=='Onset_LeanEnd')[0][0]
        stim_df = stim_df.drop(index=[Onset_LeanEnd_ind]).reset_index(drop=True)
        stim_df.loc[0, 'event'], stim_df.loc[stim_df.index[-1], 'event'] = 'Onset_Nasa', 'Offset_Nasa'
        Onset_Supine2_ind = np.where(stim_df['event']=='Onset_Supine2')[0][0]
        Offset_Supine2_ind = Onset_Supine2_ind + np.where(stim_df.loc[Onset_Supine2_ind:]['event'].str.contains('Offset_Supine'))[0][0]
        stim_df.loc[Offset_Supine2_ind, 'event'] = 'Offset_Supine2'

    return stim_df

def import_stim_data(xdf_filename:str):
    '''
    Get the stimuli dataframe from the xdf file.
    
    Args:
        xdf_filename (str): The xdf file to get the stimuli from.
    '''
    data, _ = pyxdf.load_xdf(xdf_filename, select_streams=[{'name':'StimLabels'}], verbose = False)
    stim_df = pd.DataFrame(data[0]['time_series'])
    stim_df.rename(columns={0: 'event'}, inplace=True)
    task = xdf_filename.split('/')[-1].split('task-')[-1].split('_')[0]

    # MAY NEED TO BE REFORMATTED/Have more codes/events added
    events = {
        'Onset_CPT': 200,
        'Onset_Crash': 10,
        'Offset_Crash': 11,
        'Offset_CPT': 201
    }

    stim_df['trigger'] = stim_df['event'].apply(lambda event: events[event] if event in events else np.nan)

    # COME BACK TO THESE TIME THINGS
    # relabel the event as a psychopy timestamp if the trigger is greater than 5 digits
    stim_df.loc[stim_df.trigger.astype(str).str.len() > 5, 'event'] = 'psychopy_time_stamp'
    stim_df['lsl_time_stamp'] = data[0]['time_stamps']
    stim_df['time'] = (data[0]['time_stamps'] - data[0]['time_stamps'][0])
    stim_df = reformat_events(stim_df, task=task)
    
    return stim_df

def get_event_data(event, df, stim_df):
    """
    Get the data from a given data modality that corresponds to the event in the stimuli dataframe.
    
    Args:
        event (str): The event to get the data for.
        df (pd.DataFrame): The dataframe containing the timeseries along with a column for lsl timestamps.
        stim_df (pd.DataFrame): The stimuli dataframe containing the eventa mapped to lsl timestamps.
    
    Returns:
        pd.DataFrame: The  data corresponding to the event.
        """
    if event == None:
        return df
    new_df = df.loc[(df['lsl_time_stamp'] >= stim_df.loc[stim_df['event'] == 'Onset_'+event, 'lsl_time_stamp'].values[0]) & 
                  (df['lsl_time_stamp'] <= stim_df.loc[stim_df['event'] == 'Offset_'+event, 'lsl_time_stamp'].values[0])].copy().reset_index(drop = True)
    return new_df

# get durations of certain experiment arm
def get_durations(xdf_path: str, 
    task: str, 
    stim_df: pd.DataFrame,
    df_map: dict, 
    error_map: dict
    ) -> pd.DataFrame:
    
    """
    Get the durations of each data stream and compare to their expected duration, given an experiment arm, where the expected duration is calculated from the LSL timestamps of the stimulus markers.
    
    Args:
        xdf_path (str): The path to the xdf file.
        task (str): The part of the experiment to view durations. Can be one of "Experiment", 
            "RestingState", "StoryListening", "SocialTask", or any one of the stories ('BirthdayParty', 
            'ZoomClass', 'Tornado', 'FrogDissection', 'DanceContest', 'CampFriend')
        stim_df (pd.DataFrame): The stimuli dataframe containing the events mapped to lsl timestamps.
        df_map (dict): Contains dataframes for each data modality, loaded through import_modality_data functions in utils.
        error_map (dict): Contains booleans for each data modality indicating error. 
    
    Returns:
        pd.DataFrame: The durations of each stream in seconds and mm:ss and the percent that that duration 
            comprised of the length of that experiment arm.
    """    
    streams = list(df_map.keys())

    # find expected duration (stim lsl_time_stamp length of experiment part)
    exp_start = stim_df.loc[stim_df.event == 'Onset_'+task, 'lsl_time_stamp'].values[0]
    exp_end = stim_df.loc[stim_df.event == 'Offset_'+task, 'lsl_time_stamp'].values[0]
    exp_dur = round(exp_end - exp_start, 4)

    # expected mm:ss
    exp_dt = datetime.timedelta(seconds=exp_dur)
    exp_dt_dur = str(datetime.timedelta(seconds=round(exp_dt.total_seconds())))

    # make + populate durations_df
    durations_df = pd.DataFrame(columns = ['stream', 'duration', 'mm:ss', 'percent'])
    for i, stream in enumerate(streams):
        # don't include mic in resting state
        if task == 'RestingState' and stream == 'mic':
            continue
        if error_map[stream]: 
            subject = xdf_path.split('sub-')[-1].split('_')[0]#xdf_path.split('/')[6].split('-')[1]
            print(f'No {stream} data for participant {subject}')
            continue
        # grab data for stream + experiment part
        event_data = get_event_data(task, df_map[stream], stim_df)

        # print if no data
        if event_data.empty:
            durations_df.loc[i] = [stream, 0, str(datetime.timedelta(seconds=0)), '0.0000']
            print(f'{stream} has no {task} data') 
            continue
        # calculate duration
        start = event_data['lsl_time_stamp'].values[0]
        stop = event_data['lsl_time_stamp'].values[-1]
        dur = round(stop - start, 4)

        # calculate hh:mm:ss
        dt = datetime.timedelta(seconds=dur)
        dt_dur = str(datetime.timedelta(seconds=round(dt.total_seconds())))

        # calculate percent 
        percent = round(dur/exp_dur * 100, 4)

        durations_df.loc[i] = [stream, dur, dt_dur, percent]

    # print which are short
    for i in durations_df.iterrows():
        if i[1]['duration'] == 0:
            continue
        if i[1]['duration'] < (exp_dur - 5): # 5 second margin
            print(f"{i[1]['stream']} is shorter than expected for {task} by {exp_dur - i[1]['duration']:.4f} seconds")
    
    # print + return durations_df
    durations_df.loc[durations_df.index.max() + 1] = ['expected', exp_dur, exp_dt_dur, '100.0000']
    durations_df.sort_values(by='duration', inplace=True)
    print('\n' + task + ' DataFrame')
    return durations_df

def load_xdf_from_zip(path_to_zip):  
    # Path to the tar.gz file
    tar_gz_file_path = path_to_zip # Path to the tar.gz file

    # Open the tar.gz file
    with tarfile.open(tar_gz_file_path, 'r:gz') as tar:
        file_list = tar.getnames() # List all files in the tar.gz
        file_name = [x for x in file_list if os.path.splitext(x)[1] == '.xdf'][0] # Read a specific file from the tar.gz
        file = tar.extractfile(file_name)
        file_content = file.read()
        data, info = pyxdf.load_xdf(BytesIO(file_content))
        #streams_collected = [stream['info']['name'][0] for stream in data]        
        #print(streams_collected)
    return data, info

def whole_durations(xdf_path: str, stim_df: pd.DataFrame, df_map: dict, error_map: dict) -> pd.DataFrame:
    """
    Get the durations of each data stream and compare to their expected duration, for the entire experiment, where the expected duration is 
    the max duration of any data stream.
    Args:
        xdf_path (str): The path to the xdf file.
        stim_df (pd.DataFrame): The stimuli dataframe containing the events mapped to lsl timestamps.
        df_map (dict): Contains dataframes for each data modality, loaded through import_modality_data functions in utils.
        error_map (dict): Contains booleans for each data modality indicating error. 

    Returns:
        pd.DataFrame: The durations of each stream in seconds and mm:ss and the percent that that duration comprised 
        of the max duration of all data streams. 
    """

    streams = list(df_map.keys())

    whole_durations_df = pd.DataFrame(columns = ['stream', 'duration', 'mm:ss'])
  
    # populate whole_durations_df
    for i, stream in enumerate(streams): 
        if error_map[stream]:
            subject = xdf_path.split('/')[6].split('-')[1]
            print(f'No {stream} data for participant {subject}')
            continue
        duration = df_map[stream]['lsl_time_stamp'].iloc[-1]- df_map[stream]['lsl_time_stamp'].iloc[0]
        duration = round(duration, 4)
        # convert to mm:ss
        whole_dt = datetime.timedelta(seconds=duration)
        whole_dt_dur = str(datetime.timedelta(seconds=round(whole_dt.total_seconds())))
        whole_durations_df.loc[i] = [stream, duration, whole_dt_dur]
    
    whole_durations_df.sort_values(by = 'duration', inplace = True)

    # percent
    max_dur = whole_durations_df.duration.max()
    whole_durations_df['percent'] = whole_durations_df['duration'].apply(lambda x: round(x / max_dur * 100, 4) )

    # print which are short
    for i in whole_durations_df.iterrows():
        if i[1]['duration'] == 0:
            continue
        if i[1]['duration'] < (max_dur - 30): # 30 second margin
            print(f"{i[1]['stream']} is shorter than expected by {max_dur - i[1]['duration']:.4f} seconds")

        
    whole_durations_df.sort_values(by = 'duration', inplace = True)
    return(whole_durations_df)

def get_sampling_rate(df):
    effective_sampling_rate = 1 / (df.lsl_time_stamp.diff().median())
    return effective_sampling_rate

def get_IQR(filename):
    if os.path.exists(filename) != True:
        print('File does not exist.')
    else:
        ranges_for_qc = {}
        qc_metrics = pd.read_csv(filename)
        for metric in qc_metrics.columns:
            if qc_metrics[metric].dtype in ['float64', 'int64']:
                q1 = np.percentile(qc_metrics[metric], 25)
                q3 = np.percentile(qc_metrics[metric], 75)
                ranges_for_qc.update({f'range_{metric}': [q1, q3]})
        return ranges_for_qc

# allow the functions in this script to be imported into other scripts
if __name__ == "__main__":
    pass