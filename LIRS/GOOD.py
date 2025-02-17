import json
import math
import os
import os.path as osp
import sys
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
import random
import matplotlib


from datasets.spmotif_dataset import SPMotif
from torch_geometric.data import DataLoader,Data
import ast

# dsets = ['data/SPMotif-0.4002/','data/SPMotif-0.5002/','data/SPMotif-0.5992/','data/SPMotif-0.9002/','data/SPMotif-0.4003/','data/SPMotif-0.5003/','data/SPMotif-0.5993/','data/SPMotif-0.9003/']
# dsets = ['data/SPMotif-0.400313/','data/SPMotif-0.599313/','data/SPMotif-0.900313/']
# for dat in dsets:
#     print (dat)
#     s = SPMotif(root=dat,mode='train')
#     s = SPMotif(root=dat,mode='val')
#     s = SPMotif(root=dat,mode='test')
#     print (s)
    # use data loader for s
    # data_loader = DataLoader(s, batch_size=32, shuffle=True)
    # for batch in data_loader:
    #     print (batch)
    #     print (batch.node_label.shape)
    #     print (batch.x.shape)
    #     break
    
    
    # print (s[0].node_label)
    # print (s[4000].node_label)
    # print (s[8000].node_label)



def parse_hyperparameters(file_path, file_name):
    """
    Parses the top 60 hyperparameter sets and their evaluation scores from a given text file.
    
    Args:
        file_path (str): The path to the directory containing the text file.
        file_name (str): The name of the text file to be read.
        
    Returns:
        list of dict: A list of dictionaries containing the hyperparameters and their evaluation scores.
    """
    full_path = f"{file_path}/{file_name}"
    hyperparameters_list = []

    with open(full_path, 'r') as file:
        lines = file.readlines()
        for line in lines[:1]:  # Reading only the top 60 lines
            if 'Top' in line:
                parts = line.split('Evaluation score:')
                hyperparams_str = line.split('hyperparameters: ')[1].split(', Evaluation score')[0]
                evaluation_score = float(parts[1].strip())
                # Convert string representation of dictionary to actual dictionary
                hyperparams_dict = ast.literal_eval(hyperparams_str)
                
                # Add the evaluation score to the dictionary
                hyperparams_dict['evaluation_score'] = evaluation_score
                print (hyperparams_dict)
                hyperparameters_list.append(hyperparams_dict)
    
    return hyperparameters_list


hyper_path = f"hyper_search_res/drugood_lbap_core_ec50_scaffold/"
hyper_combinations = parse_hyperparameters(hyper_path, "top_hyperparameters.txt")