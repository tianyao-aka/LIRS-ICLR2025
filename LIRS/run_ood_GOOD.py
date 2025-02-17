
from math import gamma
import time
import torch
import os.path as osp
import GCL.losses as L
import GCL.augmentors as A
from torch import nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.optim import Adam
from GCL.eval import get_split, SVMEvaluator
# from GCL.models import DualBranchContrast
from infomax import train
from model.DualBranchContrast import DualBranchContrast
from torch_geometric.nn import GCNConv, global_add_pool
from torch_geometric.data import DataLoader,Data
from torch_geometric.datasets import TUDataset

from model.utils import calc_cluster_labels
from model.ood_learner import ModelTrainer
from model.IntraClassClusteringV2 import ClusterProcessor
from model.data_utils import CustomDataset,DatasetWithSpuRep,get_indices_and_boolean_tensor
from model.utils import save_numpy_array_to_file,save_tensors_to_file
from termcolor import colored
# from utils.trainingUtils import * 
# from utils.functionalUtils import *

from copy import deepcopy
from collections import defaultdict

import argparse
import os
import sys
import warnings
import random
from datasets import spmotif_dataset
from torch_geometric.data import Batch


from GOOD.data.good_datasets.good_cmnist import GOODCMNIST
from GOOD.data.good_datasets.good_motif import GOODMotif
from GOOD.data.good_datasets.good_hiv import GOODHIV
from datasets.graphss2_dataset import get_dataloader_per, get_dataset
from ogb.graphproppred import Evaluator, PygGraphPropPredDataset
from torch_geometric.transforms import BaseTransform
import argparse

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Model Trainer Arguments')
parser.add_argument('--dataset', default='drugood_lbap_core_ec50_scaffold', type=str)
parser.add_argument('--root', default='./data', type=str, help='directory for datasets.')

parser.add_argument('--nfeat', type=int, default=3, help='Number of features.')
parser.add_argument('--nhid', type=int, default=128, help='Number of hidden units.')
parser.add_argument('--nclass', type=int, default=3, help='Number of classes.')
parser.add_argument('--nlayers', type=int, default=4, help='Number of layers.')
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate.')
parser.add_argument('--edge_dim', type=int, default=-1, help='Edge dimension.')
parser.add_argument('--save_mem', type=bool, default=True, help='Flag to save memory.')
parser.add_argument('--jk', type=str, default='last', help='Jumping knowledge method.')
parser.add_argument('--node_cls', type=bool, default=False, help='Node classification flag.')
parser.add_argument('--pooling', type=str, default='mean', help='Pooling method.')
parser.add_argument('--with_bn',action='store_true', default=False, help='Batch normalization flag.')
parser.add_argument('--weight_decay', type=float, default=5e-6, help='Weight decay for optimizer.')
parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate.')
parser.add_argument('--batch_size', type=int, default=64, help='Learning rate.')
parser.add_argument('--epochs', type=int, default=50, help='Learning rate.')
parser.add_argument('--patience', type=int, default=50, help='Patience for early stopping.')
parser.add_argument('--early_stop_epochs', type=int, default=50, help='Early stopping epochs.')
parser.add_argument('--hsic_penalty', type=float, default=0.0, help='HSIC penalty.')
parser.add_argument('--intra_cluster_labels', action='store_true', default=False, help='create labels in each class with 2 clusters')
parser.add_argument('--raw_labels', action='store_true', default=False)
parser.add_argument('--intra_cluster_penalty', type=float, default=0.0, help='cluster label penalty.')
parser.add_argument('--project_layer_num', type=int, default=2, help='Number of project layers.')
parser.add_argument('--base_gnn', type=str, default='gin', help='Base GNN model.')
parser.add_argument('--kernel_method', type=str, default='rbf', help='Kernel method for HSIC.')
parser.add_argument('--sigma', type=float, default=0.1, help='Sigma value for HSIC.')
parser.add_argument('--gamma', type=float, default=0.9, help='coef for sample reweighting in intra-cluster labelling')
parser.add_argument('--balance_sampler', action='store_true', default=False, help='balanced sampler')
parser.add_argument('--valid_metric', type=str, default='acc', help='Validation metric.')
parser.add_argument('--num_clusters', type=int, default=3, help='#clusters to use for balance sampling')

parser.add_argument('--domain', type=str, default='scaffold')
parser.add_argument('--shift', type=str, default='covariate')
parser.add_argument('--mol_encoder', action='store_true', default=False)
parser.add_argument('--vGIN', action='store_true', default=False)

parser.add_argument('--use_lr_scheduler', action='store_true', default=False)

parser.add_argument('--device', type=int, default=0, help='Device to run the model on.')
parser.add_argument('--spu_emb_path', type=str, default='', help='load spu emb')
parser.add_argument('--ood_path', type=str, default='ood_res')
parser.add_argument('--fname_str', type=str, default='', help='additional name for folder name')
parser.add_argument('--save_test_rep', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=1) 
args = parser.parse_args()


domain = args.domain
shift = args.shift

seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



def count_env_labels_by_target(dataset):
    """
    Calculate the counts of different environment labels (env_id) in each target label (y).

    Args:
        dataset (Dataset): The dataset containing the Data objects.

    Returns:
        dict: A dictionary where each key is a target label (y) and the value is another dictionary.
              This inner dictionary has keys as environment labels (env_id) and values as their counts.
    """
    # Initialize a defaultdict to store counts of env_ids for each target y
    env_counts_by_target = defaultdict(lambda: defaultdict(int))

    # Iterate over each data point in the dataset
    for data in dataset:
        y = int(data.y.item())  # Get the target label
        env_id = int(data.env_id.item())  # Get the environment label

        # Increment the count for the corresponding y and env_id
        env_counts_by_target[y][env_id] += 1

    # Convert the defaultdict to a regular dict for easier handling
    return {y: dict(env_counts) for y, env_counts in env_counts_by_target.items()}

def size_split_idx(dataset, mode='ls'):

    num_graphs = len(dataset)
    num_val   = int(0.1 * num_graphs)
    num_test  = int(0.1 * num_graphs)
    num_train = num_graphs - num_test - num_val

    num_node_list = []
    train_idx = []

    for data in dataset:
        num_node_list.append(data.num_nodes)

    sort_list = np.argsort(num_node_list)

    if mode == 'ls':
        train_idx = sort_list[2 * num_val:]
        valid_test_idx = sort_list[:2 * num_val]
    else:
        train_idx = sort_list[:-2 * num_val]
        valid_test_idx = sort_list[-2 * num_val:]
    random.shuffle(valid_test_idx)
    valid_idx = valid_test_idx[:num_val]
    test_idx = valid_test_idx[num_val:]

    split_idx = {'train': torch.tensor(train_idx, dtype = torch.long), 
                 'valid': torch.tensor(valid_idx, dtype = torch.long), 
                 'test': torch.tensor(test_idx, dtype = torch.long)}
    return split_idx

def write_results_to_file(fpath, n, s):
    # Check if the directory exists, if not, create it
    # fpath: 存放路径
    # n: 文件名 （.txt结尾）
    # s: 内容
    if not os.path.exists(fpath):
        try:
            os.makedirs(fpath)
        except:
            pass

    # Construct full file path
    full_path = os.path.join(fpath, n)

    # Open the file in write mode, which will create the file if it does not exist
    # and overwrite it if it does. Then write the string to the file.
    with open(full_path, 'w') as f:
        f.write(s)


result_fname = f"{args.dataset}/{domain}/{shift}/"
if 'goodmotif' in args.dataset.lower():
    result_fname += f"basegnn_{args.base_gnn}_withbn_{args.with_bn}_nlayers_{args.nlayers}_pooling_{args.pooling}_edgeDim_{args.edge_dim}_es_{args.early_stop_epochs}_penalty_{args.hsic_penalty}_IntraClusterPenalty_{args.intra_cluster_penalty}_gamma_{args.gamma}_kernel_{args.kernel_method}_sigma_{args.sigma}_bs_{args.balance_sampler}_numCluster_{args.num_clusters}_seed_{args.seed}/"
    result_dir = os.path.join(args.ood_path, result_fname, args.fname_str)
else:
    result_fname += f"basegnn_{args.base_gnn}_withbn_{args.with_bn}_nlayers_{args.nlayers}_pooling_{args.pooling}_edgeDim_{args.edge_dim}_es_{args.early_stop_epochs}_penalty_{args.hsic_penalty}_IntraClusterPenalty_{args.intra_cluster_penalty}_gamma_{args.gamma}_kernel_{args.kernel_method}_sigma_{args.sigma}_bs_{args.balance_sampler}_numCluster_{args.num_clusters}_seed_{args.seed}/"
    result_dir = os.path.join(args.ood_path, result_fname, args.fname_str)
args = vars(args)

print (result_dir)
if os.path.exists(result_dir):
    print (colored(f"Directory {result_dir} exists. Exiting.",'red'))
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


dataset_name = args['dataset']
if 'motif' in dataset_name.lower():
    args['nclass'] = 3
    args["nfeat"] = 1
    metric_name = 'acc'
    # args['with_bn']
    args['dropout'] = 0.
    args['vGIN']=False
    args['mol_encoder']=False
    args['edge_dim']=-1


if 'cmnist' in dataset_name.lower():
    args['nclass'] = 10
    args["nfeat"] = 3
    args['with_bn']=True
    args['dropout'] = 0.5
    args['vGIN']=True
    args['mol_encoder']=False
    args['edge_dim']=-1

if 'hiv' in dataset_name.lower():
    args['nclass'] = 2
    args["nfeat"] = 1
    args['edge_dim']=1
    if domain=='size':
        args['with_bn']=True
        args['dropout'] = 0.5
        args['vGIN']=False
        args['mol_encoder']=True
    else:
        args['with_bn']=True
        args['dropout'] = 0.
        args['vGIN']=True
        args['mol_encoder']=True

if 'ogbg' in dataset_name.lower():
    args['nclass'] = 2
    args["nfeat"] = 1
    args['edge_dim']=1
    if 'bbbp' in dataset_name.lower():
        if domain=='size':
            args['with_bn']=True
            args['dropout'] = 0.
            args['vGIN']=False
            args['mol_encoder']=True
        if domain=='scaffold':  
            args['with_bn']=True
            args['dropout'] = 0.5
            args['vGIN']=True
            args['mol_encoder']=True
    elif "bace" in dataset_name.lower():
        if domain=='size':
            args['with_bn']=True
            args['dropout'] = 0.
            args['vGIN']=False
            args['mol_encoder']=True
        if domain=='scaffold':  
            args['with_bn']=False
            args['dropout'] = 0.0
            args['vGIN']=False
            args['mol_encoder']=True

#! init dataset

domain = args['domain']
shift = args['shift']

workers = 1 if torch.cuda.is_available() else 0


if "motif" in args['dataset'].lower():
    dataset, meta_info = GOODMotif.load("GOOD_data/", domain=domain, shift=shift, generate=False)
    train_dataset = dataset['train']
    val_dataset = dataset['val']
    test_dataset = dataset['test']
    
    #! load spurious emb
    path = args['spu_emb_path']
    print (colored(f'spu_emb path:{path}','yellow'))
    if len(path)>4:
        spu_emb = torch.load(path)
    else:
        print (colored("Spu Emb is None. Cautious!",'yellow'))
        spu_emb = None
    c_labels = None
    binary_c_id = None
    binary_c_counts = None
    intra_cluster_logits =None
    intra_cluster_labels = None
    sample_weights = None
    if args['balance_sampler']:
        print (colored('Use balance sampler','red'))
        spu_emb_np = np.asarray(spu_emb.cpu())
        Y = np.asarray([train_dataset[i].y.item() for i in range(len(train_dataset))])
        c_labels = calc_cluster_labels(spu_emb_np,Y,num_classes=3,num_clusters=3)

    if args['intra_cluster_labels']:
        print (colored('Use intra-class clustering labels','blue'))
        Y = torch.tensor([train_dataset[i].y.item() for i in range(len(train_dataset))])
        cluster_process = ClusterProcessor(spu_emb,Y,num_classes=3, num_clusters = args['num_clusters'],gamma=args['gamma'])
        intra_cluster_logits,intra_cluster_labels,sample_weights = cluster_process.process()

    if args['raw_labels']:
        args['num_clusters'] = 3
        print (colored('Use raw labels for spurious logits generation','blue'))
        Y = torch.tensor([train_dataset[i].y.item() for i in range(len(train_dataset))])
        cluster_process = ClusterProcessor(spu_emb,Y,num_classes=3, num_clusters = 3,gamma=args['gamma'])
        intra_cluster_logits,intra_cluster_labels,sample_weights = cluster_process.process_no_cluster()   #! no clustering is performed

    train_data_list = [train_dataset[i] for i in range(len(train_dataset))]
    train_dataset = DatasetWithSpuRep(train_data_list,dataset_name='spmotif',spu_rep=spu_emb,cluster_id=c_labels,intra_cluster_pred_logits=intra_cluster_logits, intra_cluster_labels = intra_cluster_labels,sample_weights = sample_weights)

    if args['balance_sampler']:
        train_data_list = [train_dataset[i] for i in range(len(train_dataset))]
        train_loader = create_custom_loader(train_data_list,batch_size=args["batch_size"])

    else:
        train_loader = DataLoader(train_dataset,batch_size=args["batch_size"],num_workers=workers,shuffle=True)
    
    valid_loader = DataLoader(val_dataset,batch_size=args["batch_size"],shuffle=False,num_workers=workers)
    test_loader = DataLoader(test_dataset,batch_size=args["batch_size"],shuffle=False,num_workers=workers)
    args['nclass'] = 3
    metric_name='acc'
    args['valid_metric'] = metric_name

elif args['dataset'].lower().startswith('ogbg'):
    #drugood_lbap_core_ic50_assay.json
    metric_name='auc'
    args['valid_metric'] = metric_name
    dataset = PygGraphPropPredDataset(root=args["root"], name=args["dataset"])
    if domain == "scaffold":
        split_idx = dataset.get_idx_split()
    else:
        split_idx = size_split_idx(dataset,mode='nls')  #! be cautious here!
    train_dataset = dataset[split_idx["train"]]
    val_dataset = dataset[split_idx["valid"]]
    test_dataset = dataset[split_idx["test"]]
    
    #! load spurious emb
    path = args['spu_emb_path']
    print (colored(f'spu_emb path:{path}','yellow'))
    if len(path)>4:
        spu_emb = torch.load(path)
    else:
        print (colored("Spu Emb is None. Cautious!",'yellow'))
        spu_emb = None
    c_labels = None
    binary_c_id = None
    binary_c_counts = None
    if args['balance_sampler']:
        print (colored('Use balance sampler','red'))
        spu_emb_np = np.asarray(spu_emb.cpu())
        Y = np.asarray([train_dataset[i].y.item() for i in range(len(train_dataset))])
        s = time.time()
        print ('start clustering')
        c_labels = calc_cluster_labels(spu_emb_np,Y,num_classes=2,num_clusters=args['num_clusters'])
        t = time.time()-s
        print (colored(f'calc cluster labels time:{t}','yellow'))


    if args['intra_cluster_labels']:
        print (colored('Use intra-class clustering labels','blue'))
        Y = torch.tensor([train_dataset[i].y.item() for i in range(len(train_dataset))])
        cluster_process = ClusterProcessor(spu_emb,Y,num_classes=2, num_clusters = args['num_clusters'],gamma=args['gamma'])
        intra_cluster_logits,intra_cluster_labels,sample_weights = cluster_process.process()
    
    if args['raw_labels']:
        args['num_clusters'] = 2
        print (colored('Use raw labels for spurious logits generation','blue'))
        Y = torch.tensor([train_dataset[i].y.item() for i in range(len(train_dataset))])
        cluster_process = ClusterProcessor(spu_emb,Y,num_classes=2, num_clusters = 3,gamma=args['gamma'])
        intra_cluster_logits,intra_cluster_labels,sample_weights = cluster_process.process_no_cluster()   #! no clustering is performed

    train_data_list = [train_dataset[i] for i in range(len(train_dataset))]
    train_dataset = DatasetWithSpuRep(train_data_list,dataset_name='drugood',spu_rep=spu_emb,cluster_id=c_labels,intra_cluster_pred_logits=intra_cluster_logits, intra_cluster_labels = intra_cluster_labels,sample_weights = sample_weights)

    if args['balance_sampler']:
        train_data_list = [train_dataset[i] for i in range(len(train_dataset))]
        train_loader = create_custom_loader(train_data_list,batch_size=args["batch_size"])

    else:
        train_loader = DataLoader(train_dataset,batch_size=args["batch_size"],num_workers=workers,shuffle=True)


    # train_loader = DataLoader(train_dataset, batch_size=args["batch_size"], shuffle=True, num_workers=workers)
    valid_loader = DataLoader(val_dataset, batch_size=args["batch_size"], shuffle=False,num_workers=workers)
    test_loader = DataLoader(test_dataset, batch_size=args["batch_size"], shuffle=False,num_workers=workers)
    test_loader_single_data = DataLoader(test_dataset, batch_size=1, shuffle=False,num_workers=workers)
    print ('len of test dataset of:',args['dataset'])
    print (len(test_dataset))


if "goodhiv" in args['dataset'].lower():
    dataset, meta_info = GOODHIV.load(args['root'], domain=args['domain'], shift=shift, generate=False)
    train_dataset = dataset['train']
    val_dataset = dataset['val']
    test_dataset = dataset['test']
    #! load spurious emb
    path = args['spu_emb_path']
    print (colored(f'spu_emb path:{path}','yellow'))
    if len(path)>4:
        spu_emb = torch.load(path)
    else:
        print (colored("Spu Emb is None. Cautious!",'yellow'))
        spu_emb = None
    c_labels = None
    binary_c_id = None
    binary_c_counts = None
    intra_cluster_logits =None
    intra_cluster_labels = None
    sample_weights = None
    if args['balance_sampler']:
        print (colored('Use balance sampler','red'))
        spu_emb_np = np.asarray(spu_emb.cpu())
        Y = np.asarray([train_dataset[i].y.item() for i in range(len(train_dataset))])
        c_labels = calc_cluster_labels(spu_emb_np,Y,num_classes=3,num_clusters=3)

    if args['intra_cluster_labels']:
        print (colored('Use intra-class clustering labels','blue'))
        Y = torch.tensor([train_dataset[i].y.item() for i in range(len(train_dataset))])
        cluster_process = ClusterProcessor(spu_emb,Y,num_classes=2, num_clusters = args['num_clusters'],gamma=args['gamma'])
        intra_cluster_logits,intra_cluster_labels,sample_weights = cluster_process.process()

    if args['raw_labels']:
        args['num_clusters'] = 2
        print (colored('Use raw labels for spurious logits generation','blue'))
        Y = torch.tensor([train_dataset[i].y.item() for i in range(len(train_dataset))])
        cluster_process = ClusterProcessor(spu_emb,Y,num_classes=2, num_clusters = args['num_clusters'],gamma=args['gamma'])
        intra_cluster_logits,intra_cluster_labels,sample_weights = cluster_process.process_no_cluster()   #! no clustering is performed

    train_data_list = [train_dataset[i] for i in range(len(train_dataset))]
    train_dataset = DatasetWithSpuRep(train_data_list,dataset_name='goodhiv',spu_rep=spu_emb,cluster_id=c_labels,intra_cluster_pred_logits=intra_cluster_logits, intra_cluster_labels = intra_cluster_labels,sample_weights = sample_weights)

    if args['balance_sampler']:
        train_data_list = [train_dataset[i] for i in range(len(train_dataset))]
        train_loader = create_custom_loader(train_data_list,batch_size=args["batch_size"])

    else:
        train_loader = DataLoader(train_dataset,batch_size=args["batch_size"],num_workers=workers,shuffle=True)
    
    valid_loader = DataLoader(val_dataset,batch_size=args["batch_size"],shuffle=False,num_workers=workers)
    test_loader = DataLoader(test_dataset,batch_size=args["batch_size"],shuffle=False,num_workers=workers)
    args['nclass'] = 2
    metric_name='auc'
    args['valid_metric'] = metric_name


args['device'] = 'cpu' if not torch.cuda.is_available() else args['device']
model = ModelTrainer(**args)
print ('fit model')
model.fit(train_loader,valid_loader,test_loader,epochs=args["epochs"])
model.load_state_dict(model.best_states)
res = model.valid_metric_list
if "motif" in args['dataset'] and domain=='basis':
    res = res[50:]
res = sorted(res,key = lambda x:x[0],reverse=True)
val_score,test_score = res[0]
res = np.array([val_score,test_score])


if args["save_test_rep"]:
    model.eval()
    reps = []
    y = []
    with torch.no_grad():
        for dat in test_loader:
            data = dat.to(args["device"])
            g = model.gnn(data.x, data.edge_index, batch=data.batch,edge_attr = data.edge_attr,return_both_rep=False)
            reps.append(g.cpu())
            y.append(data.y.cpu())
    reps = torch.cat(reps,dim=0).numpy()
    y = torch.cat(y,dim=0).numpy()
    test_reps = [reps,y]
    save_tensors_to_file(test_reps,result_dir,"test_reps_with_Y")


print ('best res:',res)

save_numpy_array_to_file(res,result_dir,"val_test_metric")

torch.cuda.empty_cache()
sys.stdout.flush()
sys.stderr.flush()
os._exit(0)
