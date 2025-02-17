import sys
sys.path.append('../src')

import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
from pathlib import Path
from gsat import GSAT, ExtractorMLP
from utils import get_data_loaders, get_model, set_seed, Criterion, init_metric_dict, load_checkpoint,topk_nodes_from_attn,remove_topk_edges,fidelity
from trainer import run_one_epoch, update_best_epoch_res, get_viz_idx, visualize_results

from torch_geometric.loader import DataLoader
from utils import process_data, get_preds, save_checkpoint
from torch_geometric.explain.metric import groundtruth_metrics
from torchmetrics import AUROC,Accuracy
from torchmetrics.classification import BinaryF1Score
from tqdm import tqdm
from trainer import eval_one_batch

from datetime import datetime
import argparse
import os
import time

# provide a parser for the command line
parser = argparse.ArgumentParser()
# add augument for string arguments


def calculate_means(A: torch.Tensor, B: torch.Tensor, labels: torch.Tensor):
    """
    Given two tensors A and B of shape (N,), both representing probabilities,
    and a label tensor of shape (N,) with values in {0, 1}, this function calculates
    the mean ratio of B/A for correctly predicted 0 samples and A/B for correctly 
    predicted 1 samples using 0.5 as the threshold.

    Args:
    A (torch.Tensor): Tensor of probabilities with shape (N,).
    B (torch.Tensor): Tensor of probabilities with shape (N,).
    labels (torch.Tensor): Tensor of labels with shape (N,), values in {0, 1}.

    Returns:
    mean_0 (float): Mean ratio of B/A for correctly predicted 0 samples.
    mean_1 (float): Mean ratio of A/B for correctly predicted 1 samples.
    """

    # Apply threshold to get predictions
    A = A.detach().cpu()
    B = B.detach().cpu()
    labels = labels.detach().cpu()
    pred_A = (A > 0.5).float()
    
    # Get indices of correctly predicted samples for 0 and 1
    correct_pred_0 = (pred_A == labels) & (labels == 0)
    correct_pred_1 = (pred_A == labels) & (labels == 1)
    
    # Calculate B[idx]/A[idx] for correctly predicted 0 samples
    ratios_0 = B[correct_pred_0] / A[correct_pred_0]
    mean_0 = ratios_0.mean().item() if len(ratios_0) > 0 else 0.0
    
    # Calculate A[idx]/B[idx] for correctly predicted 1 samples
    ratios_1 = A[correct_pred_1] / B[correct_pred_1]
    mean_1 = ratios_1.mean().item() if len(ratios_1) > 0 else 0.0
    return mean_0, mean_1



parser.add_argument('--dataset', default='graph-sst2', type=str)
parser.add_argument('--root', default='./data', type=str, help='directory for datasets.')
parser.add_argument('--layer', default=4, type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--device', default=0, type=int)

#! for GOOD datasets
parser.add_argument('--domain', default='scaffold', type=str)
parser.add_argument('--shift', default='covariate', type=str)

args = parser.parse_args()

dataset_name = args.dataset
model_name = 'GIN'
domain = args.domain
shift = args.shift

method_name = 'GSAT'
cuda_id = args.device
seed = 0
set_seed(seed)


data_dir = Path('../data')
config_dir = Path('../config')
device = torch.device(f'cuda:{cuda_id}' if cuda_id >= 0 else 'cpu')


if model_name == 'GIN':
    model_config = {'model_name': 'GIN', 'hidden_size': 64, 'n_layers': args.layer, 'dropout_p': 0.5, 'use_edge_attr': True}
else:
    assert model_name == 'PNA'
    model_config = {'model_name': 'PNA', 'hidden_size': 80, 'n_layers': args.layer, 'dropout_p': 0.3, 'use_edge_attr': False, 
                    'atom_encoder': True, 'aggregators': ['mean', 'min', 'max', 'std'], 'scalers': False}


metric_dict = deepcopy(init_metric_dict)
model_dir = data_dir / dataset_name / 'logs' / (datetime.now().strftime("%m_%d_%Y-%H_%M_%S") + '-' + dataset_name + '-' + model_name + '-seed' + str(seed) + '-' + method_name)


batch_size = args.batch_size
loaders, test_set, x_dim, edge_attr_dim, num_class, aux_info = get_data_loaders(data_dir, dataset_name, config_dir = config_dir, batch_size=batch_size, random_state=seed,
                                                                                splits={'train': 0.8, 'valid': 0.1, 'test': 0.1}, 
                                                                                mutag_x=True if dataset_name == 'mutag' else False,domain=args.domain,shift=args.shift)
model_config['deg'] = aux_info['deg']
if 'ogbg' in dataset_name or 'hiv' in dataset_name:
    model_config['atom_encoder'] = True
    if 'bace' in dataset_name or 'bbbp' in dataset_name:
        model_config['hidden_size'] = 128
        # args.epochs=150
    if 'hiv' in dataset_name:
        model_config['hidden_size'] = 300


# print (len(loaders['train'])*batch_size,len(loaders['valid'])*batch_size,len(loaders['test'])*batch_size)


clf = get_model(x_dim, edge_attr_dim, num_class, aux_info['multi_label'], model_config, device)

extractor = ExtractorMLP(model_config['hidden_size'], learn_edge_att=False).to(device)
optimizer = torch.optim.Adam(list(extractor.parameters()) + list(clf.parameters()), lr=1e-3, weight_decay=3.0e-6)
criterion = Criterion(num_class, aux_info['multi_label'])
gsat = GSAT(clf, extractor, criterion, optimizer, learn_edge_att=False, final_r=0.7)


if 'drugood' in dataset_name.lower() or 'ogbg' in dataset_name.lower():
    auc_func = AUROC(task='binary').to(gsat.device)   #! default be to auc. change if not
if 'sst' in dataset_name.lower():
    auc_func = Accuracy(task='binary').to(gsat.device)
if 'twitter' in dataset_name.lower():
    auc_func = Accuracy(task='multiclass',num_classes=3).to(gsat.device)
if 'hiv' in dataset_name.lower():
    auc_func = AUROC(task='binary').to(gsat.device)
if 'ogbg' in dataset_name.lower():
    auc_func = AUROC(task='binary').to(gsat.device)
if 'cmnist' in dataset_name.lower():
    auc_func = Accuracy(task='multiclass',num_classes=num_class).to(gsat.device)

# train_loader_single = loaders['entire_train']
# node_avg = []
# for d in train_loader_single:
#     print (d.y)
#     break
# # print (sum(node_avg)/len(node_avg))
# sys.exit(0)
print ('num_classes is:',num_class)
time_start = time.time()

for epoch in range(args.epochs):
    train_res = run_one_epoch(gsat, loaders['train'], epoch, 'train', dataset_name, seed, model_config['use_edge_attr'], aux_info['multi_label'],metric_func = auc_func)
    valid_res = run_one_epoch(gsat, loaders['valid'], epoch, 'valid', dataset_name, seed, model_config['use_edge_attr'], aux_info['multi_label'],metric_func = auc_func)
    # test_res = run_one_epoch(gsat, loaders['test'], epoch, 'test', dataset_name, seed, model_config['use_edge_attr'], aux_info['multi_label'],metric_func = auc_func)
    
    # metric_dict = update_best_epoch_res(gsat, train_res, valid_res, test_res, metric_dict, dataset_name, epoch, model_dir)
    # print(f'[Seed {seed}, Epoch: {epoch}]: Best Epoch: {metric_dict["metric/best_clf_epoch"]}, '
    #       f'Best Val Pred ACC/ROC: {metric_dict["metric/best_clf_valid"]:.3f}, Best Test Pred ACC/ROC: {metric_dict["metric/best_clf_test"]:.3f}, '
    #       f'Best Test X AUROC: {metric_dict["metric/best_x_roc_test"]:.3f}')
    # print('='*50)
    # print('='*50)

time_end = time.time()
print(f'Training time: {time_end - time_start:.2f}s')
sys.exit()
best_epoch = metric_dict['metric/best_clf_epoch']
load_checkpoint(gsat, model_dir, model_name=f'gsat_epoch_{best_epoch}', map_location=device)

train_loader_single = loaders['entire_train']

if num_class>=3:
    auc_func = AUROC(task='multiclass',num_classes=num_class).to(gsat.device)
else:
    auc_func = AUROC(task='binary').to(gsat.device)

loader_single = loaders['entire_train']
topK=[4,8,12]

infos = []
node_list_dict_top4 = {} #key is topK
node_list_dict_top8 = {} #key is topK
node_list_dict_top12 = {} #key is topK
for t in topK:
    gt = []
    y_hat_preds = []
    y_hat_removal_preds = [] 
    node_list_with_mask = []
    cnt = 0
    for batch in tqdm(loader_single):
            # cnt +=1
            # if cnt>100: break
            batch = batch.to(gsat.device)
            att, loss_dict, clf_logits = eval_one_batch(gsat,batch,500)
            if batch.x.shape[0]>t:
                nodes = topk_nodes_from_attn(att,batch.edge_index,t)
            else:
                # print ('cautious:',batch.x.shape[0],t)
                nodes = topk_nodes_from_attn(att,batch.edge_index,1)
                node_mask_false = torch.zeros(1)
                node_list_with_mask.append([nodes,node_mask_false,node_mask_false,node_mask_false])
                continue

            node_mask_true = torch.ones(len(nodes))
            node_mask_false = torch.zeros(len(nodes))
            if num_class<=2:
                y = batch.y.view(-1,)
                y_hat = torch.sigmoid(clf_logits).detach().cpu().view(-1,)
                new_edge_index,new_edge_attr = remove_topk_edges(att,batch,t)
                batch.edge_index = new_edge_index
                batch.edge_attr = new_edge_attr
                att, loss_dict, clf_logits = eval_one_batch(gsat,batch,500)
                y_hat_removal = torch.sigmoid(clf_logits).detach().cpu().view(-1,)
                if np.random.uniform()>0.99:
                    print ('show info:', y.item(),y_hat.item(),y_hat_removal.item())
                if y.item()==0:
                    if y_hat_removal.item()<=y_hat.item():
                        node_list_with_mask.append([nodes,node_mask_false,node_mask_false,node_mask_false])
                    else:
                        t_ = [nodes]
                        if y_hat_removal.item()>=y_hat.item()+0.1:
                            t_ += [node_mask_true]
                        else:
                            t_ += [node_mask_false]
                        
                        if y_hat_removal.item()>=y_hat.item()+0.2:
                            t_ += [node_mask_true]
                        else:
                            t_ += [node_mask_false]
                        if y_hat_removal.item()>=y_hat.item()+0.3:
                            t_ += [node_mask_true]
                        else:
                            t_ += [node_mask_false]
                        node_list_with_mask.append(t_)
                
                if y.item()==1:
                    if y_hat_removal.item()>=y_hat.item():
                        node_list_with_mask.append([nodes,node_mask_false,node_mask_false,node_mask_false])
                    else:
                        t_ = [nodes]
                        if y_hat_removal.item()<=y_hat.item()-0.1:
                            t_ += [node_mask_true]
                        else:
                            t_ += [node_mask_false]
                        
                        if y_hat_removal.item()<=y_hat.item()-0.2:
                            t_ += [node_mask_true]
                        else:
                            t_ += [node_mask_false]
                        if y_hat_removal.item()<=y_hat.item()-0.3:
                            t_ += [node_mask_true]
                        else:
                            t_ += [node_mask_false]
                        node_list_with_mask.append(t_)

                gt.append(y)
                y_hat_preds.append(y_hat)
                y_hat_removal_preds.append(y_hat_removal)
            else:
                #! need to modify for instance level node select
                y = batch.y.view(-1,)
                y_hat = torch.softmax(clf_logits,dim=-1).detach().cpu().view(1,-1)
                new_edge_index,new_edge_attr = remove_topk_edges(att,batch,t)
                batch.edge_index = new_edge_index
                batch.edge_attr = new_edge_attr
                att, loss_dict, clf_logits = eval_one_batch(gsat,batch,500)
                y_hat_removal = torch.softmax(clf_logits,dim=-1).detach().cpu().view(1,-1)
                # print (y,y_hat,y_hat_removal)
                y_true = y.item()
                y_hat_true_prob = y_hat[0, y_true].item()
                y_hat_removal_true_prob = y_hat_removal[0, y_true].item()
                t_ = [nodes]
                
                if y_hat_removal_true_prob <= y_hat_true_prob - 0.1:
                    t_ += [node_mask_true]
                else:
                    t_ += [node_mask_false]
                
                if y_hat_removal_true_prob <= y_hat_true_prob - 0.2:
                    t_ += [node_mask_true]
                else:
                    t_ += [node_mask_false]
                
                if y_hat_removal_true_prob <= y_hat_true_prob - 0.3:
                    t_ += [node_mask_true]
                else:
                    t_ += [node_mask_false]
                
                node_list_with_mask.append(t_)
                gt.append(y)
                y_hat_preds.append(y_hat)
                y_hat_removal_preds.append(y_hat_removal)
                

    gt = torch.cat(gt,dim=0)
    y_hat_preds = torch.cat(y_hat_preds,dim=0)
    y_hat_removal_preds = torch.cat(y_hat_removal_preds,dim=0)
    # gt0_delta_score,gt1_delta_score = calculate_means(y_hat_preds,y_hat_removal_preds,gt)
    print('Groundtruth Metrics')
    auc1 = auc_func(y_hat_preds.to(gsat.device),gt.to(gsat.device))
    auc2 = auc_func(y_hat_removal_preds.to(gsat.device),gt.to(gsat.device))
    fid = fidelity(gt.cpu(),y_hat_preds.cpu(),y_hat_removal_preds.cpu(),binary=True if num_class==2 else False)
    print ("auc, auc after removal and fidelity are:",auc1.cpu(),auc2.cpu(),fid)
    # print ('delta change of probs for labels 0 and 1:',gt0_delta_score,gt1_delta_score)
    key = t
    info =f"auc1:{auc1},auc2:{auc2},auc-gap:{auc1-auc2},fidelity:{fid}."
    infos.append(info)
    if t==4:
        node_list_dict_top4[t] = (node_list_with_mask,info)
    if t==8:
        node_list_dict_top8[t] = (node_list_with_mask,info)
    if t==12:
        node_list_dict_top12[t] = (node_list_with_mask,info)
    print (f"TopK:{t} Done")

fpath = f"../explain_res/{dataset_name}/{domain}/{shift}/{model_name}_layer_{args.layer}"

print (infos)

# if not exists, mkdir for path
if not os.path.exists(fpath):
    os.makedirs(fpath)

torch.save(node_list_dict_top4, os.path.join(fpath,'res_dict_top4.pt'))
torch.save(node_list_dict_top8, os.path.join(fpath,'res_dict_top8.pt'))
torch.save(node_list_dict_top12, os.path.join(fpath,'res_dict_top12.pt'))
