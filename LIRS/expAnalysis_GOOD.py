import os
import torch
import numpy as np
import pandas as pd
import re
import traceback 
import argparse


parser = argparse.ArgumentParser(description='Model Trainer Arguments')
parser.add_argument('--dataset', default='GOODMOTIF', type=str)
parser.add_argument('--domain', default='basis', type=str, help='directory for datasets.')
parser.add_argument('--shift', default='covariate', type=str, help='directory for datasets.')
args = parser.parse_args()


def select_test_acc(valid_acc, test_acc, k=5):
    """
    Select the test accuracy based on the epoch at which valid_acc[i] / std(valid_acc[i-k:i]) is the best.

    Args:
        valid_acc (list or np.array): A list or array of validation accuracies.
        test_acc (list or np.array): A list or array of test accuracies corresponding to the validation accuracies.
        k (int): The window size for calculating the standard deviation.

    Returns:
        float: The selected test accuracy corresponding to the best ratio of valid_acc[i] / std(valid_acc[i-k:i]).
    """
    best_ratio = -np.inf
    best_epoch = -1
    
    for i in range(k, len(valid_acc)):
        # Calculate the standard deviation over the window [i-k, i)
        window_std = np.std(valid_acc[i-k:i])
        # Avoid division by zero
        if window_std > 0:
            ratio = valid_acc[i] / window_std
            if ratio > best_ratio:
                best_ratio = ratio
                best_epoch = i
    
    # Return the corresponding test accuracy for the best epoch
    return test_acc[best_epoch] if best_epoch != -1 else 0.

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
    Groups the dataframe by specified columns, excluding 'seed', 'val_score', and 'test_score',
    and calculates the average and standard deviation for val_score and test_score.
    
    Parameters:
    df (pd.DataFrame): Input dataframe with specified columns.

    Returns:
    pd.DataFrame: Aggregated dataframe with mean and std of val_score and test_score.
    """
    # Get all columns and exclude 'seed', 'val_score', and 'test_score'
    groupby_cols = [col for col in df.columns if col not in ['seed', 'val_score', 'test_score']]

    # Perform the groupby and aggregation
    grouped_df = df.groupby(groupby_cols).agg(
        val_score_mean=('val_score', 'mean'),
        val_score_std=('val_score', 'std'),
        test_score_mean=('test_score', 'mean'),
        test_score_std=('test_score', 'std'),
        num_seed=('val_score', 'count'),
        list_vals = ('test_score', lambda x: list(x)),
    ).reset_index()
    return grouped_df

def process_directory(dir_path,name = None,domain = None,shift = None, extra_name = None):
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
            dirs[:] = [d for d in dirs if 'bs_' in d]
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
                        # print (val_test_metric.shape)
                        if len(val_test_metric.shape)>1 and domain=='size':
                            # print (111)
                            res = sorted(val_test_metric,key = lambda x:x[0],reverse=True)
                            val_score, test_score = res[0]
                        if len(val_test_metric.shape)>1 and domain=='basis':
                            # print (222)
                            val_res = [i[0] for i in val_test_metric]
                            test_res = [i[1] for i in val_test_metric]
                            test1 = select_test_acc(val_res,test_res,k=3)
                            test2 = select_test_acc(val_res,test_res,k=5)
                            val_score, test_score = test1, test2
                        if len(val_test_metric.shape)==1:
                            # print (333)
                            val_score, test_score = val_test_metric
                        # Parse the first-level subdirectory name
                        config_1 = parse_config_string(subdir)
                        # Parse the second-level subdirectory name
                        if "epoch_epoch_" in sub_subdir:
                            sub_subdir = sub_subdir.replace("epoch_epoch_","epoch_")
                        
                        config_2 = parse_config_string(sub_subdir)
                        config = {**config_1, **config_2}
                        config['val_score'] = val_score
                        config['test_score'] = test_score
                        print (config)
                        
                        all_data.append(config)


    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    df = aggregate_scores(df)
    # dir = "excel_formal/edgedim0_mean/gridsearch_unbiased/"
    # dir = "excel_formal/intraCluster_v2/"
    # dir = "excel_formal/balance_sampling/"
    
    dir = "excel_results/GOOD/EQUAD/"
    # dir = "excel_formal/wo_intra_cluster/"
    # dir = "excel_formal/bs_v2/"
    if not os.path.exists(dir):
        print ('create dir')
        os.makedirs(dir)
    if name is not None:
        if 'spmotif' in name.lower():
            df.to_csv(f"{dir}/{name}_{domain}_{shift}_results_deltaAcc.csv",index=False)
        else:
            if extra_name is not None:
                df.to_csv(f"{dir}/{name}_{domain}_{shift}_{extra_name}_results.csv",index=False)
            else:
                df.to_csv(f"{dir}/{name}_{domain}_{shift}_results.csv",index=False)
    return df


dset = ["SPMotif-0.4003","SPMotif-0.5993","SPMotif-0.9003","SPMotif-0.4002","SPMotif-0.5992","SPMotif-0.9002","SPMotif-0.4001","SPMotif-0.5991","SPMotif-0.9001"]
dset = ["SPMotif-0.4002","SPMotif-0.5992","SPMotif-0.9002","SPMotif-0.4002","SPMotif-0.5992","SPMotif-0.9002","SPMotif-0.4003","SPMotif-0.5993","SPMotif-0.9003","SPMotif-0.400313","SPMotif-0.599313","SPMotif-0.900313"]
dset = ["drugood_lbap_core_ec50_scaffold","drugood_lbap_core_ec50_assay",
        "drugood_lbap_core_ki_size","drugood_lbap_core_ki_scaffold","drugood_lbap_core_ki_assay"]
# dset = ["SPMotif-0.40033","SPMotif-0.59933","SPMotif-0.80033","SPMotif-0.90033"]
dset = [args.dataset]
domain = args.domain
shift=args.shift
extra_name = "LIRS_size_reversed"
# dset = ["drugood_lbap_core_ec50_size"]


for d in dset:
    print (d)
    dir_path = f'ood_res_GOOD/intraCluster_molbace_0908/{d}/{domain}/{shift}/'
    dir_path = f'ood_res_GOOD/EQUAD/{d}/{d}/{domain}/{shift}/'
    dir_path = f'ood_res_GOOD/EQUAD/{d}/reversed/{d}/{domain}/{shift}/'
    dir_path = f'ood_res_GOOD/intra_cluster_bace_reversed/{d}/{domain}/{shift}/'
    # dir_path = f'ood_res_GOOD/ablation_motif_no_biased_infomax/{d}/{domain}/{shift}/'
    # dir_path = f'ood_res_GOOD/ablation_motif_no_intracluster/{d}/{domain}/{shift}/'
    # dir_path = f'ood_res_GOOD/GOODHIV_no_intracluster/{d}/{domain}/{shift}/'
    # dir_path = f'ood_res_GOOD/GOODHIV_no_biased_infomax/{d}/{domain}/{shift}/'
    # dir_path = f'ood_res_formal/afterHyper/intra_cluster/{d}/'
    df = process_directory(dir_path,name = d,domain=domain,shift=shift,extra_name = extra_name)


# dset = ["drugood_lbap_core_ec50_assay","drugood_lbap_core_ec50_scaffold","drugood_lbap_core_ec50_size"]
# dset = ["drugood_lbap_core_ki_assay","drugood_lbap_core_ki_scaffold","drugood_lbap_core_ki_size"]
# dset = ["drugood_lbap_core_ki_size"]
# dset = ["drugood_lbap_core_ic50_assay"]


# for d in dset:
#     try:
#         print (d)
#         dir_path = f'ood_res_edgedim0_mean/biased_False/{d}/'
#         dir_path = f'tmp/ood_res_intra_cluster_nobias_retry/{d}/'
#         df = process_directory(dir_path,name = d)
#     except:
#         # print error reason
        
#         traceback.print_exc() 
#         pass
