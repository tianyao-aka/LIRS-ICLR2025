
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
from model.DualBranchContrast import DualBranchContrast
from torch_geometric.nn import GCNConv, global_add_pool
from torch_geometric.data import DataLoader,Data
from torch_geometric.datasets import TUDataset
from model.gin import GIN
from model.data_utils import CustomDataset,get_indices_and_boolean_tensor
from termcolor import colored
from ogb.graphproppred import Evaluator, PygGraphPropPredDataset
# from utils.trainingUtils import * 
# from utils.functionalUtils import *
import time
from copy import deepcopy
import argparse
import os
import sys
import warnings
import random
from datasets import spmotif_dataset
from torch_geometric.data import Batch

from datasets.drugood_dataset import DrugOOD
from datasets.graphss2_dataset import get_dataloader_per, get_dataset
from datasets.mnistsp_dataset import CMNIST75sp
from datasets.spmotif_dataset import SPMotif
from torch_geometric.transforms import BaseTransform
from drugood.datasets import build_dataset
from mmcv import Config

from GOOD.data.good_datasets.good_cmnist import GOODCMNIST
from GOOD.data.good_datasets.good_motif import GOODMotif
from GOOD.data.good_datasets.good_hiv import GOODHIV

warnings.filterwarnings("ignore")

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
        

def size_split_idx(dataset, mode='ls'):

    num_graphs = len(dataset)
    num_val   = int(0.1 * num_graphs)
    num_test  = int(0.1 * num_graphs)
    num_train = num_graphs - num_test - num_val

    num_node_list = []
    train_idx = []
    valtest_list = []

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



class FeatureSelector(BaseTransform):
    def __init__(self):
        pass

    def __call__(self, data):
        """
        Concatenate random features to each node in the graph. If node features
        do not exist, new random features are assigned as node features.

        Parameters:
        - data (torch_geometric.data.Data): The graph data object.

        Returns:
        - torch_geometric.data.Data: The modified graph data object with added features.
        """
        x = data.x[:,:4]
        data.x = x
        return data

def save_tensor_to_file(X, fpath, name):
    """
    Save a PyTorch tensor X to a specified file path and name.
    
    Parameters:
    - X (torch.Tensor): Tensor of shape (N, D) to be saved.
    - fpath (str): The directory where the tensor should be saved.
    - name (str): The name of the file to save the tensor as.
    
    Returns:
    None
    """
    
    # Create the directory if it doesn't exist
    if not os.path.exists(fpath):
        os.makedirs(fpath)
    
    # Full path to the file
    full_path = os.path.join(fpath, name)
    
    # Remove the file if it already exists
    if os.path.exists(full_path):
        os.remove(full_path)
    
    # Save the tensor to the file
    torch.save(X, full_path)



def adjust_node_indices(batch_node_indices, node_indices_lengths, batch_ptr):
    """
    Adjust the node indices with offsets using lengths of node indices and batch pointers.

    Parameters:
    batch_node_indices (torch.Tensor): Tensor containing node indices of the batch.
    node_indices_lengths (torch.Tensor): Tensor containing the lengths of node indices for each graph in the batch.
    batch_ptr (torch.Tensor): Tensor containing pointers to the start of each graph in the batch.

    Returns:
    torch.Tensor: Adjusted node indices with appropriate offsets.
    """
    adjusted_node_indices = []
    start_idx = 0

    for i, length in enumerate(node_indices_lengths):
        end_idx = start_idx + length
        indices = batch_node_indices[start_idx:end_idx]
        offset = batch_ptr[i]
        adjusted_indices = indices + offset
        adjusted_node_indices.append(adjusted_indices)
        start_idx = end_idx

    return torch.cat(adjusted_node_indices)


class FC(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(FC, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
            nn.ReLU()
        )
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x) + self.linear(x)



class Encoder(torch.nn.Module):
    def __init__(self, gcn1, gcn2, mlp1, mlp2, aug1, aug2):
        super(Encoder, self).__init__()
        self.gcn1 = gcn1
        self.gcn2 = gcn2
        self.mlp1 = mlp1
        self.mlp2 = mlp2
        self.aug1 = aug1
        self.aug2 = aug2

    def forward(self,data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x1, edge_index1, edge_weight1 = self.aug1(x, edge_index)
        x2, edge_index2, edge_weight2 = self.aug2(x, edge_index)
        z1, g1 = self.gcn1(x1, edge_index1, batch=batch,edge_attr=None,return_both_rep=True)
        z2, g2 = self.gcn2(x2, edge_index2, batch=batch,edge_attr=None,return_both_rep=True)
        h1, h2 = [self.mlp1(h) for h in [z1, z2]]
        g1, g2 = [self.mlp2(g) for g in [g1, g2]]
        return h1, h2, g1, g2


def train(encoder_model, contrast_model, dataloader, optimizer):
    encoder_model.train()
    epoch_loss = 0
    for data in dataloader:
        data.adj_node_indices = adjust_node_indices(data.node_indices, data.node_indices_len, data.ptr)
        data = data.to(device) if torch.cuda.is_available() else data.to('cpu')
        optimizer.zero_grad()
        if data.x is None:
            num_nodes = data.batch.size(0)
            data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)
    
        h1, h2, g1, g2 = encoder_model(data)
        loss = contrast_model(h1=h1, h2=h2, g1=g1, g2=g2, batch=data.batch,data=data) #! new argument data for adjust pos masks
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss


def test(encoder_model, dataloader,numPoints = 10,num_trials=15):
    results = []
    encoder_model.eval()
    x = []
    y = []
    with torch.no_grad():
        for data in dataloader:
            data = data.to(device) if torch.cuda.is_available() else data.to('cpu')
            if data.x is None:
                num_nodes = data.batch.size(0)
                data.x = torch.ones((num_nodes, 1), dtype=torch.float32, device=data.batch.device)
            _, _, g1, g2 = encoder_model(data)
            x.append(g1 + g2)
            y.append(data.y)
    x = torch.cat(x, dim=0)
    y = torch.cat(y, dim=0)
    x_np = x.cpu().numpy()
    y_np = y.cpu().numpy()
    kmeans_acc_result = []
    #! use linear-SVM as augmentation
    # msg = train_and_evaluate_svm(x_np,y_np,K=numPoints,S=int(0.6*numPoints),num_trials=num_trials)
    return None,x.cpu()


if __name__ == '__main__':
    seed=1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # provide a parser for the command line
    parser = argparse.ArgumentParser()
    # add augument for string arguments

    parser.add_argument('--hidden_dims',type=int,default=64)
    parser.add_argument('--batch_size',type=int,default=64)
    parser.add_argument('--num_layers',type=int,default=4)
    parser.add_argument('--edge_dim',type=int,default=-1) # -1 means not using edge attr
    parser.add_argument('--biased', action='store_true', default=False, help='bias the infomax')
    parser.add_argument('--ignore', action='store_true', default=False, help='how to bias infomax')
    parser.add_argument('--dataset',type=str,default="Graph-SST5")
    parser.add_argument('--root', default='./data', type=str, help='directory for datasets.')
    parser.add_argument('--SSL',type=str,default="MVGRL")
    parser.add_argument('--explain_res_layer',type=int,default=5)
    parser.add_argument('--explain_res_topK',type=int,default=8) 
    parser.add_argument('--explain_res_thres_index',type=int,default=0)  # 0-2: thres = {0.1,0.2,0.3}
    parser.add_argument('--device_id',type=int,default=0)
    parser.add_argument('--epochs',type=int,default=100)
    parser.add_argument('--domain',type=str,default='scaffold')
    parser.add_argument('--shift',type=str,default='covariate')
    parser.add_argument('--seed',type=int,default=1)
    # parser.add_argument('--fold_index',type=int,default=0)
    
    args = parser.parse_args()
    args = vars(args)
    domain = args['domain']
    shift = args['shift']
    if 'motif' not in args['dataset'].lower():
        K = args['explain_res_topK']
        thres_index = args['explain_res_thres_index']
        if "ogbg" in args['dataset'].lower():
            name = args['dataset'].lower().replace("-","_")
        else:
            name = args['dataset'].lower()
        if args['shift']=='covariate':
            if not os.path.exists(f"explain_res/{args['dataset'].lower()}/{args['domain']}/covariate/"):
                node_indices_path = f"explain_res/{args['dataset'].lower()}/{args['domain']}/GIN_layer_{args['explain_res_layer']}/res_dict_top{K}.pt"
            else:
                node_indices_path = f"explain_res/{args['dataset'].lower()}/{args['domain']}/covariate/GIN_layer_{args['explain_res_layer']}/res_dict_top{K}.pt"
        else:
            node_indices_path = f"explain_res/{args['dataset'].lower()}/{args['domain']}/concept/GIN_layer_{args['explain_res_layer']}/res_dict_top{K}.pt"
        
        res = torch.load(node_indices_path,map_location=torch.device('cpu'))
        res = res[K][0]
        node_indices = [i[0] for i in res]
        node_masks = [i[thres_index+1] for i in res]
    
    
    if not args['biased']:
        embeddingWritingPath = f"experiment_results/SSL_embedding/{args['dataset']}/{args['domain']}/{args['shift']}/biased_infomax_{args['biased']}/hidden_dims_{args['hidden_dims']}_num_layers_{args['num_layers']}/"
    else:
        if 'motif' in args['dataset'].lower():
            embeddingWritingPath = f"experiment_results/SSL_embedding/{args['dataset']}/{args['domain']}/{args['shift']}/biased_infomax_{args['biased']}_ignore_{args['ignore']}/hidden_dims_{args['hidden_dims']}_num_layers_{args['num_layers']}/"
        else:
            embeddingWritingPath = f"experiment_results/SSL_embedding/{args['dataset']}/{args['domain']}/{args['shift']}/biased_infomax_{args['biased']}_ignore_{args['ignore']}/hidden_dims_{args['hidden_dims']}_num_layers_{args['num_layers']}/expLayer_{args['explain_res_layer']}_expTopK_{args['explain_res_topK']}_expThres_{args['explain_res_thres_index']}/"
    
    if os.path.exists(embeddingWritingPath):
        sys.exit(f'------------------------already finished running-------------------------')
    print ('emb path:',embeddingWritingPath)
    
    hidden_dim = args["hidden_dims"]
    num_layers = args["num_layers"]
    dataset_name = args["dataset"]
    device = torch.device(f'cuda:{args["device_id"]}') if torch.cuda.is_available() else torch.device('cpu')
    num_workers = 1 if torch.cuda.is_available() else 0
    print (colored(f"using device {device}",'red','on_white'))
    path = "GOOD_data/"
    mol_encoder = False
    
    if "goodhiv" in args['dataset'].lower():
        #drugood_lbap_core_ic50_assay.json
        args['nclass'] = 2
        metric_name = 'auc' #! watch out this!
        args['valid_metric'] = metric_name
        mol_encoder = True
        dataset, meta_info = GOODHIV.load(args['root'], domain=args['domain'], shift=shift, generate=False)
        train_dataset = dataset["train"]
        train_data_list = [train_dataset[i] for i in range(len(train_dataset))]
        custom_train_dataset = CustomDataset(train_data_list,dataset_name='goodhiv',node_indices=node_indices,node_mask=node_masks)
        train_loader = DataLoader(custom_train_dataset,batch_size=args["batch_size"],num_workers=num_workers,shuffle=True)
        full_train_loader = DataLoader(train_dataset, batch_size=args["batch_size"], shuffle=False,num_workers=num_workers)
        val_dataset = dataset["val"]
        valid_loader = DataLoader(val_dataset, batch_size=args["batch_size"], shuffle=False,num_workers=num_workers)
        edge_dim = 1
        num_classes = 2
        input_dim = 1
    
    
    elif args["dataset"].lower().startswith('ogbg'):
        dataset = PygGraphPropPredDataset(root=args["root"], name=args["dataset"])
        input_dim = 1
        args['nclass'] = 2
        mol_encoder = True
        metric_name = 'auc' #! watch out this!
        args['valid_metric'] = metric_name
        if domain == "scaffold":
            split_idx = dataset.get_idx_split()
        else:
            split_idx = size_split_idx(dataset)
        train_dataset = dataset[split_idx["train"]]
        val_dataset = dataset[split_idx["valid"]]
        
        full_train_loader = DataLoader(train_dataset, batch_size=args["batch_size"], shuffle=False,num_workers=num_workers)
        #! process data
        train_data_list = [train_dataset[i] for i in range(len(train_dataset))]
        custom_train_dataset = CustomDataset(train_data_list,dataset_name=args['dataset'],node_indices=node_indices,node_mask=node_masks)
        train_loader = DataLoader(custom_train_dataset,batch_size=args["batch_size"],num_workers=num_workers,shuffle=True)
        input_dim = train_dataset[0].x.shape[1]
        edge_dim = 1
        num_classes = 2
    
    
    elif "cmnist" in args['dataset'].lower():
        #drugood_lbap_core_ic50_assay.json
        args['nclass'] = 10
        num_classes = 10
        input_dim = 3
        edge_dim=-1
        metric_name = 'acc' #! watch out this!
        args['valid_metric'] = metric_name
        dataset, meta_info = GOODCMNIST.load("GOOD_data/", domain='color', shift='covariate', generate=False)
        train_dataset = dataset["train"]
        train_data_list = [train_dataset[i] for i in range(len(train_dataset))]
        custom_train_dataset = CustomDataset(train_data_list,dataset_name='goodcmnist',node_indices=node_indices,node_mask=node_masks)
        train_loader = DataLoader(custom_train_dataset,batch_size=args["batch_size"],num_workers=num_workers,shuffle=True)
        full_train_loader = DataLoader(train_dataset, batch_size=args["batch_size"], shuffle=False,num_workers=num_workers)
        val_dataset = dataset["val"]
        valid_loader = DataLoader(val_dataset, batch_size=args["batch_size"], shuffle=False,num_workers=num_workers)
        

    elif 'motif' in args['dataset'].lower() and 'spmotif' not in args['dataset'].lower():
        dataset, meta_info = GOODMotif.load("GOOD_data/", domain=domain, shift=shift, generate=False)
        train_dataset = dataset['train']
        val_dataset = dataset['val']
        test_dataset = dataset['test']
        full_train_loader = DataLoader(train_dataset, batch_size=args["batch_size"], shuffle=False,num_workers=num_workers)
        # val_dataset = SPMotif(root=f'data/{args["dataset"]}',mode='val')
        # test_dataset = SPMotif(root=f'data/{args["dataset"]}',mode='test')
        #! process dataset to custom dataset
        train_data_list = [train_dataset[i] for i in range(len(train_dataset))]
        custom_train_dataset = CustomDataset(train_data_list,dataset_name='goodmotif')
        train_loader = DataLoader(custom_train_dataset,batch_size=args["batch_size"],num_workers=num_workers,shuffle=True)
        # valid_loader = DataLoader(val_dataset,batch_size=args["batch_size"],shuffle=False,num_workers=num_workers)
        # test_loader = DataLoader(test_dataset,batch_size=args["batch_size"],shuffle=False,num_workers=num_workers)
        input_dim=1
        num_classes = 3
        edge_dim = -1

    elif 'spmotif' in args['dataset'].lower():
        train_dataset = SPMotif(root=f'data/{args["dataset"]}',mode='train')
        full_train_loader = DataLoader(train_dataset, batch_size=args["batch_size"], shuffle=False,num_workers=num_workers)
        # val_dataset = SPMotif(root=f'data/{args["dataset"]}',mode='val')
        # test_dataset = SPMotif(root=f'data/{args["dataset"]}',mode='test')
        #! process dataset to custom dataset
        train_data_list = [train_dataset[i] for i in range(len(train_dataset))]
        custom_train_dataset = CustomDataset(train_data_list,dataset_name='spmotif')

        train_loader = DataLoader(custom_train_dataset,batch_size=args["batch_size"],num_workers=num_workers,shuffle=True)
        # valid_loader = DataLoader(val_dataset,batch_size=args["batch_size"],shuffle=False,num_workers=num_workers)
        # test_loader = DataLoader(test_dataset,batch_size=args["batch_size"],shuffle=False,num_workers=num_workers)
        input_dim=4
        num_classes = 3

        #! debug
        # dat = next(iter(train_loader))
        # print (dat)
        # print (dat.node_label)
        # print (dat.node_indices)
        # print (dat.node_indices_len)
        # print (dat.node_mask)
        # print (dat.ptr)
        # adj_node_indices = adjust_node_indices(dat.node_indices, dat.node_indices_len, dat.ptr)
        # dat.adj_node_indices = adj_node_indices
        # print (dat)
        # sys.exit()


    elif args["dataset"].lower() in ['graph-sst5','graph-sst2']:
        dataset = get_dataset(dataset_dir=args["root"], dataset_name=args["dataset"], task=None)
        dataloader,train_dataset,val_dataset,test_dataset\
                = get_dataloader_per(dataset, batch_size=args["batch_size"], small_to_large=True, seed=args["seed"],return_set=True)
        train_loader = dataloader['train']
        valid_loader = dataloader['eval']
        test_loader = dataloader['test']
        full_train_loader = DataLoader(train_dataset, batch_size=args["batch_size"], shuffle=False)
        input_dim = 768
        num_classes = int(args["dataset"][-1].lower()) if args["dataset"][-1].lower() in ['2', '5'] else 3
        
    elif args["dataset"].lower() in ['graph-twitter']:
        dataset = get_dataset(dataset_dir=args["root"], dataset_name=args["dataset"], task=None)
        dataloader,train_dataset,val_dataset,test_dataset \
        = get_dataloader_per(dataset, batch_size=args["batch_size"], small_to_large=False, seed=args["seed"],return_set=True)
        train_loader = dataloader['train']
        valid_loader = dataloader['eval']
        test_loader = dataloader['test']
        full_train_loader = DataLoader(train_dataset, batch_size=args["batch_size"], shuffle=False)
        input_dim = 768
        num_classes = int(args["dataset"][-1].lower()) if args["dataset"][-1].lower() in ['2', '5'] else 3


    aug1 = A.Identity()
    aug2 = A.Identity()
    aug2 = A.PPRDiffusion(alpha=0.2, use_cache=False)
    gin1 = GIN(nfeat=input_dim, nhid=hidden_dim, nclass=num_classes, nlayers=num_layers,edge_dim= edge_dim, dropout=0.0, jk='last',node_cls=False,pooling='sum',dataset=args["dataset"].lower(),mol_encoder = mol_encoder).to(device)
    gin2 = GIN(nfeat=input_dim, nhid=hidden_dim, nclass=num_classes, nlayers=num_layers,edge_dim= edge_dim, dropout=0.0, jk='last',node_cls=False,pooling='sum',dataset=args["dataset"].lower(), mol_encoder = mol_encoder).to(device)
    mlp1 = FC(input_dim=hidden_dim, output_dim=hidden_dim)
    mlp2 = FC(input_dim=hidden_dim, output_dim=hidden_dim)
    encoder_model = Encoder(gcn1=gin1, gcn2=gin2, mlp1=mlp1, mlp2=mlp2, aug1=aug1, aug2=aug2).to(device)
    contrast_model = DualBranchContrast(loss=L.JSD(), mode='G2L',biased=args['biased'],device=device,ignore=args['ignore']).to(device)
    optimizer = Adam(encoder_model.parameters(), lr=1e-3)

    
    #! use loss as the metric to select model
    min_loss = 10000.0
    tr_emb_dict = {}
    time_start = time.time()
    with tqdm(total=args['epochs'], desc='(T)') as pbar:
        for epoch in range(1, args['epochs']+1):
            loss = train(encoder_model, contrast_model, train_loader, optimizer)
            pbar.set_postfix({'loss': loss})
            pbar.update()
            if epoch%10==0 and epoch>0:
                _,tr_emb = test(best_model, full_train_loader,num_trials=15,numPoints=10)
                tr_emb_dict[f'epoch_{epoch}_loss_{loss}']=tr_emb
            if epoch<10 and epoch>3:
                _,tr_emb = test(best_model, full_train_loader,num_trials=15,numPoints=10)
                tr_emb_dict[f'epoch_{epoch}_loss_{loss}']=tr_emb
            
            if loss < min_loss:
                min_loss = loss
                best_epoch = epoch
                best_model = deepcopy(encoder_model)
        _,tr_emb = test(best_model, full_train_loader,num_trials=15,numPoints=10)
        tr_emb_dict[f'epoch_{best_epoch}_loss_{min_loss}']=tr_emb
    
    # ! use 10-fold to evaluate the performance of linear SVM
    time_end = time.time()
    time_cost = time_end - time_start
    
    write_results_to_file(f"timeCalc/{args['dataset']}_{args['domain']}/seed_{args['seed']}",'result_time.txt',f"time cost:{time_cost}")
    
    # _,tr_emb = test(best_model, full_train_loader,num_trials=15,numPoints=10)
    # _,val_emb = test(best_model, valid_loader,num_trials=15,numPoints=10)
    # _,test_emb = test(best_model, test_loader,num_trials=15,numPoints=10)
    # write_results_to_file(resultWritingPath,'result.txt',msg)
    
    # for k in tr_emb_dict:
    #     save_tensor_to_file(tr_emb_dict[k].cpu(),embeddingWritingPath,f'graph_emb_{k}.pt')
    # save_tensor_to_file(val_emb,embeddingWritingPath,'val_graph_embedding.pt')
    # save_tensor_to_file(test_emb,embeddingWritingPath,'test_graph_embedding.pt')
    # print ('success')
    
    # print (colored(f"test_results:{test_result}",'blue','on_white'))
    
    
    
    