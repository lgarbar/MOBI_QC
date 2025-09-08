import pyxdf
import pandas as pd
import numpy as np
from glob import glob
import datetime
import re
import matplotlib.pyplot as plt
from utils import *
import sys
import argparse

def et_val(et_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the percentage of valid data for all data columns (excluding time + validity columns) in the eye-tracking data.
    Args:
        et_df (pd.DataFrame): Dataframe containing the eye-tracking data.
    Returns:
        val_df (pd.DataFrame): Dataframe containing the percentage of valid data for each variable.
    """
    # percent valid for all data columns (excluding time + validity columns)

    # remove columns w validity or time data 
    time_cols = et_df.filter(like = 'time').columns
    val_cols = et_df.filter(like = 'validity').columns
    qc_cols = time_cols.append(val_cols)
    et_data_cols = et_df.columns.drop(qc_cols)

    # percent non-NaN for each variable
    val_df = pd.DataFrame(columns= ['variable', 'percent_valid'])
    val_df['variable'] = et_data_cols

    for i, var in enumerate(et_data_cols):
        val_df.loc[i, 'percent_valid'] = 1 - et_df[var].isna().mean()

    return val_df

def et_invalid_data(et_df: pd.DataFrame) -> tuple[float, float, float, float, float, float]:
    """
    Calculate the percentage of invalid data for each eye tracking variables.
    Args:
        et_df (pd.DataFrame): Dataframe containing the eye tracking data.
    Returns:
        left_gaze_point_invalid, right_gaze_point_invalid, left_gaze_origin_invalid, right_gaze_origin_invalid, left_pupil_invalid, right_pupil_invalid (floats): Decimals representing amount of data invalid from each variable.
    """
    et_df['invalid'] = 0
    et_df['blink_filt'] = 0
    et_df['off_screen_filt'] = 0

    # masks
    if 'blink_confidence' in et_df:
        blink_mask = et_df['blink_confidence'].apply(
            lambda data: (data is not None and len(data) >= 2 and (data[0] > 0.5 or data[1] > 0.5))
        )
        et_df.loc[blink_mask, 'blink_filt'] = 1
        et_df.loc[blink_mask, 'invalid'] = 1
    
    if 'ET3S_scene_number' in et_df:
        offscreen_mask = et_df['ET3S_scene_number'] != 0
        et_df.loc[offscreen_mask, 'off_screen_filt'] = 1
        et_df.loc[offscreen_mask, 'invalid'] = 1

    blink_filt = et_df[et_df['blink_filt'] == 1]
    off_screen_filt = et_df[et_df['off_screen_filt'] == 1]
    invalid_filt = et_df[et_df['invalid'] == 1]
    et_df_filt = et_df[et_df['invalid'] == 0]

    blink_perc = np.round((len(blink_filt) / len(et_df)) * 100, 2)
    off_screen_perc = np.round((len(off_screen_filt) / len(et_df)) * 100, 2)
    invalid_perc = np.round((len(invalid_filt) / len(et_df)) * 100, 2)
    valid_perc = np.round((len(et_df_filt) / len(et_df)) * 100, 2)

    return et_df, blink_perc, off_screen_perc, invalid_perc, valid_perc

def xyz_measures_check(val_df: pd.DataFrame) -> bool:
    """
    Check if all x, y, z coordinates have the same percentage of validity within each measure (within each LR, within each gaze point/origin/diameter).
    Args:
        val_df (pd.DataFrame): Dataframe containing the percentage of valid data for each variable.
    Returns:
        val_flag1 (bool): True if all coordinates have the same percentage of validity, False otherwise.
    """
    # all coordinates have the same % validity within each measure (LR, gaze point/origin/diameter)
    # compare coordinates (0,1,2) (validity within measures)
    val_flag1 = True
    for i in range(1, len(val_df)):
        # get variables that end in numbers 
        root = re.sub(r"_\d+$", "", val_df.loc[i, 'variable']) # remove the number from the var name. ie 'left_gaze_origin_in_user_coordinate_system_1' becomes 'left_gaze_origin_in_user_coordinate_system'
        if root in val_df.loc[i-1, 'variable']: # if the var root is the same as the previous var root
            current_percent = val_df.loc[i, 'percent_valid']
            prev_percent = val_df.loc[i-1, 'percent_valid']
            if current_percent != prev_percent:
                print("ERROR: {} does not equal {}!".format(val_df.loc[i-1, 'variable'], val_df.loc[i, 'variable']))
                val_flag1 = False

    return val_flag1

def coordinate_system_check(val_df: pd.DataFrame) -> bool:
    """
    Check if the percentage of NaNs is the same between coordinate systems (UCS and TBCS for gaze origin, and between UCS and display area for gaze point).
    Args:
        val_df (pd.DataFrame): Dataframe containing the percentage of valid data for each variable.
    Returns:
        val_flag2 (bool): True if the percentage of NaNs is the same between coordinate systems, False otherwise.
    """
    # validity between coordinate systems 
    # all coordinates have the same % validity within each measure (LR, gaze point/origin/diameter)

    val_flag2 = True
    for i in range(1, len(val_df)):
        root = val_df.loc[i, 'variable'].split("_in")[0] # ie 'left_gaze_origin'
        root_percent = val_df.loc[i, 'percent_valid']

        matching = val_df[val_df['variable'].str.contains(root)] # df with all variables containing 'left_gaze_origin'

        for i in matching.index:
                matching_percent = matching.loc[i, 'percent_valid']
                matching_variable = matching.loc[i, 'variable']

                if root_percent != matching_percent: # compare root percent to all percents in matching df 
                    print("ERROR: {} and {} were different by a difference of {}.".format(matching_variable, root, (root_percent-matching_percent)))
                    val_flag2 = False

    return val_flag2

def pupil_validity_stats(et_df_filt: pd.DataFrame) -> dict:
    """
    Compute statistics on valid pupil data for left and right eyes.
    
    Args:
        et_df_filt (pd.DataFrame): Filtered eyetracking DataFrame with 'pupil_left' and 'pupil_right' columns.
        
    Returns:
        dict: Contains valid proportions and absolute difference between eyes.
    """
    left_valid_prop = len(et_df_filt[et_df_filt['pupil_left'] > 0]) / len(et_df_filt)
    right_valid_prop = len(et_df_filt[et_df_filt['pupil_right'] > 0]) / len(et_df_filt)
    abs_diff = abs(left_valid_prop - right_valid_prop)
    
    stats = {
        'left_valid_prop': left_valid_prop,
        'right_valid_prop': right_valid_prop,
        'abs_diff': abs_diff,
        'n_total': len(et_df_filt),
        'n_left_valid': len(et_df_filt[et_df_filt['pupil_left'] > 0]),
        'n_right_valid': len(et_df_filt[et_df_filt['pupil_right'] > 0])
    }
    
    return stats


def et_qc(filename: str, stim_df: pd.DataFrame, event=None) -> tuple[dict, pd.DataFrame, str]:
    """
    Main function to extract eye tracking quality control metrics.
    Args:
        filename (str): Path to the XDF file containing eye-tracking data.
        stim_df (pd.DataFrame): dataframe containing stimulus markers.
        task (str): arm of the experiment for which user wants quality control performed.
    Returns:
        vars (dict): Dictionary containing quality control metrics.
        et_df (pd.DataFrame): Dataframe containing eye tracking data.
        et_error (bool): Whether there was an error loading eye tracking data.

    """
    sub_id = filename.split('sub-')[1].split('/')[0]
    vars = {}
    vars['event'], vars['sampling_rate'], vars['left_gaze_point_invalid'], vars['right_gaze_point_invalid'], vars['left_gaze_origin_invalid'], vars['right_gaze_origin_invalid'], vars['left_pupil_invalid'], vars['right_pupil_invalid'], vars['xyz_measures_check'], vars['coordinate_system_check'], vars['LR_mean_diff'], vars['percent_over02'] = np.zeros(12)

    # try:
    whole_et_df = import_et_data(filename)
    if not stim_df:
        stim_df = import_stim_data(filename)
    et_df = get_event_data(event = event, df = whole_et_df, stim_df = stim_df)

    sampling_rate = get_sampling_rate(et_df)
    vars['event'] = event
    vars['sampling_rate'] = sampling_rate
    print(f"Effective sampling rate: {sampling_rate:.4f}")

    et_df, vars['blink_perc'], vars['off_screen_perc'], vars['invalid_perc'], vars['valid_perc'] = et_invalid_data(et_df)
    val_df = et_df[et_df['invalid']==0]
    print(
        f"{vars['blink_perc']}% of data detected as blinks. "
        f"{vars['off_screen_perc']}% of data detected as off screen. "
        f"{vars['invalid_perc']}% of data invalid. "
        f"{vars['valid_perc']}% of data valid."
    )
    vars['pupil_stats'] = pupil_validity_stats(val_df)
    print(f"Pupil stats: {vars['pupil_stats']}")
    et_error = None

    return vars, et_df, et_error
    # # if et_nums is empty
    # except ZeroDivisionError: 
    #     vars['percent_over02'] = float('nan')
    #     et_error = 'missing_data'        
    #     print(f'Error: Significant amount of ET data missing for participant {sub_id} in {filename}.')
    #     return vars, whole_et_df, et_error
    # except: # leaving this without a specific error for now because we have no PTs without ET data!
        
    #     whole_et_df = pd.DataFrame()
    #     vars.update({key: float('nan') for key in vars.keys()})
    #     et_error = 'no_data'
    #     print(f'Error: No ET data found for participant {sub_id} in {filename}.')
    #     return vars, whole_et_df, et_error

# allow the functions in this script to be imported into other scripts
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ET QC script")
    parser.add_argument("filename", help="Path to the XDF file")
    parser.add_argument("stim_fpath", nargs="?", default=None, help="Optional path to stim file (.csv or .parquet)")
    parser.add_argument("--event", default=None, help="Optional event name to override default")

    args = parser.parse_args()

    filename = args.filename
    stim_fpath = args.stim_fpath
    event_arg = args.event

    if not os.path.exists(filename):
        print(f"Error: file not found -> {filename}")
        sys.exit(1)

    if stim_fpath:
        if not os.path.exists(stim_fpath):
            print(f"Error: file not found -> {stim_fpath}")
            sys.exit(1)

        try:
            if stim_fpath.endswith('.csv'):
                stim_df = pd.read_csv(stim_fpath)
            elif stim_fpath.endswith('.parquet'):
                stim_df = pd.read_parquet(stim_fpath, engine='fastparquet')
            else:
                print("Error: stim file must be .csv or .parquet")
                sys.exit(1)
        except Exception as e:
            print(f"Error reading stim file: {e}")
            sys.exit(1)
    else:
        stim_df = False

    default_event = load_default_event(filename)
    if default_event is None:
        print("Warning: no default event found for this task")
        event = None
    else:
        event = default_event
    if event_arg is not None:
        event = event_arg

    # try:
    et_qc(filename, stim_df, event=event)
    # except Exception as e:
    #     print(f"Error running eet_qc: {e}")
    #     sys.exit(1)