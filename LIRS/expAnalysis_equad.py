import os
import torch
import numpy as np
import pandas as pd
import re
import traceback

def parse_config_string(config_string):
    """
    Parse a configuration string into a dictionary.

    Parameters:
    config_string (str): The configuration string to parse.

    Returns:
    dict: A dictionary of configuration parameters.
    """
    
    config = {}
    parts = config_string.split('_')
    # Ensure parts are in pairs
    for i in range(0, len(parts) - 1, 2):
        key = parts[i]
        value = parts[i + 1]
        config[key] = value
    return config


def aggregate_scores(df):
    """
    Groups the dataframe by specified columns and calculates the average and standard deviation
    for val_score and test_score.
    
    Parameters:
    df (pd.DataFrame): Input dataframe with specified columns.

    Returns:
    pd.DataFrame: Aggregated dataframe with mean and std of val_score and test_score.
    """
    grouped_df = df.groupby([
        'basegnn', 'nlayers', 'es', 'penalty', 'gamma',  'biased', 'ignore', 
        'epoch'
    ]).agg(
        val_score_mean=('val_score', 'mean'),
        val_score_std=('val_score', 'std'),
        test_score_mean=('test_score', 'mean'),
        test_score_std=('test_score', 'std')
    ).reset_index()
    
    return grouped_df

def process_directory(dir_path,name = None):
    """
    Process the given directory to extract subdirectory configurations and metrics.

    Parameters:
    dir_path (str): The path to the main directory.

    Returns:
    pd.DataFrame: A DataFrame containing configurations and val/test scores.
    """
    all_data = []

    for root, dirs, files in os.walk(dir_path):
        # Filter the first-level subdirectories
        if root == dir_path:
            dirs[:] = [d for d in dirs]
        
        # Process only first-level subdirectories
        if root != dir_path:
            continue

        for subdir in dirs:
            subdir_path = os.path.join(root, subdir)
            for _, sub_subdirs, sub_files in os.walk(subdir_path):
                for sub_subdir in sub_subdirs:
                    sub_subdir_path = os.path.join(subdir_path, sub_subdir)
                    npy_file_path = os.path.join(sub_subdir_path, 'val_test_metric.npy')
                    if os.path.exists(npy_file_path):
                        val_test_metric = np.load(npy_file_path)
                        val_score, test_score = val_test_metric
                        
                        # Parse the first-level subdirectory name
                        config_1 = parse_config_string(subdir)
                        # Parse the second-level subdirectory name
                        config_2 = parse_config_string(sub_subdir)

                        config = {**config_1, **config_2}
                        config['val_score'] = val_score
                        config['test_score'] = test_score

                        all_data.append(config)


    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    df = aggregate_scores(df)
    dir = "tmp/excel/equad/edgedim0_mean/gridsearch_unbiased/"
    dir = "tmp/excel/equad/"
    if not os.path.exists(dir):
        print ('create dir')
        os.makedirs(dir)
    if name is not None:
        if 'spmotif' in name.lower():
            df.to_csv(f"{dir}/{name}_results_deltaAcc.csv",index=False)
        else:
            df.to_csv(f"{dir}/{name}_results.csv",index=False)
    return df


# dset = ["SPMotif-0.400313","SPMotif-0.599313","SPMotif-0.900313"]
# for d in dset:
#     dir_path = f'ood_res_deltaAcc/equad/{d}/'
#     df = process_directory(dir_path,name = d)



dset = ["drugood_lbap_core_ic50_assay"]
dset = ["drugood_lbap_core_ki_size"]
# dset = ["drugood_lbap_core_ic50_assay"]
for d in dset:
    try:
        print (d)
        dir_path = f'tmp/ood_res/equad/{d}/'
        df = process_directory(dir_path,name = d)
    except:
        # print error reason
        
        traceback.print_exc() 
        pass




