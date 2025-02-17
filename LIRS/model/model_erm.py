import torch.nn as nn
import torch.nn.functional as F
import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch_geometric.nn import GCNConv
from torch_sparse import coalesce, SparseTensor
import torch.optim as optim
from torch_geometric.nn import global_add_pool,global_mean_pool,AttentionalAggregation
from torch.autograd import grad
from itertools import chain
from torchviz import make_dot
import sys
from tqdm import tqdm

import os.path as osp

from torch.optim import Adam
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.data import DataLoader,Data

from torchmetrics import AUROC,Accuracy
import numpy as np
import pandas as pd
import random
import string
from termcolor import colored
from copy import deepcopy

from model.base_model import BaseModel
from model.gcn import GCN
from model.gin import GIN
from model.utils import *

# try:
#     from modelNew.base_model import BaseModel
#     from modelNew.ssl_module import Encoder
#     from modelNew.gcn import GCN
#     from modelNew.gin import GIN
#     from modelNew.utils import *
# except:
#     print ('import from local')
#     from base_model import BaseModel
#     from gcn import GCN
#     from gin import GIN
#     from ssl_module import Encoder
#     from utils import *


class Model(BaseModel):
    def __init__(self, nfeat, nhid, nclass, nlayers=2, dropout=0.5,edge_dim = -1,save_mem=True,jk='last',node_cls=False,pooling='sum',
                with_bn=False, weight_decay=5e-6,lr=1e-3,lr_scheduler=True,patience=50,early_stop_epochs=10,lr_decay=0.75,penalty=0.1,project_layer_num=2,edge_penalty=1.0,with_bias=True,base_gnn='gin',valid_metric='acc', device='cpu', **args):

        super(Model, self).__init__()
        assert device is not None, "Please specify 'device'!"
        self.rnd_id = ''.join(random.choices(string.digits, k=16))  # for caching stuffs
        self.debug = args['debug']
        self.useAutoAug = False
        self.lr = lr
        self.weight_decay = weight_decay
        self.best_states = None
        self.best_meta_model = None
        self.device = device
        self.jk = jk
        self.node_cls = node_cls
        self.edge_dim = edge_dim
        self.cls_header = self.create_mlp(nhid,nhid,nclass,project_layer_num)
        self.linear_refit_layer = self.create_mlp(nhid,nhid,nclass,project_layer_num)

        self.penalty = penalty
        
        
        #  pretraining epochs
        self.pe = args['pretraining_epochs']
        
        
        if base_gnn=='gin':
            self.gnn = GIN(nfeat, nhid, nclass, nlayers, dropout=dropout,edge_dim=edge_dim,jk=jk,node_cls=node_cls,pooling=pooling,
                with_bn=with_bn, weight_decay=weight_decay)
        

        self.gnn.to(device)
        
        # self.exclude_meta_mlp_optimizer = Adam([params for name, params in self.named_parameters() if "meta" not in name], lr=lr)
        
        self.optimizer = Adam(self.parameters(), lr=lr,weight_decay=weight_decay)
        self.ce_loss = nn.CrossEntropyLoss()
        self.metric_func = Accuracy(task='multiclass',num_classes=nclass,top_k=1).to(device) if valid_metric=='acc' else AUROC(task='binary').to(device)
        self.metric_name = valid_metric
        self.train_grad_sim = []
        self.val_grad_sim = []
        
        self.valid_metric_list = []
        self.meta_valid_metric_list = []
        self.best_valid_metric = -1.
        self.test_metric=-1.
        self.early_stop_epochs = early_stop_epochs
        self.epochs_since_improvement=0
        self.stop_training=False
        
        self.train_metrics = []
        self.val_metrics = []
        self.test_metrics = []
        self.train_grad_sim = []
        self.val_grad_sim = []
        self.test_grad_sim = []


    def create_mlp(self,input_dim, hidden_dim, output_dim, num_layers,cls=True):
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        if cls:
            layers.append(nn.Linear(hidden_dim, output_dim))
        return nn.Sequential(*layers)
    
    
    def forward(self,x,edge_index,batch):
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        batch = batch.to(self.device)
        g = self.gnn(x, edge_index, batch=batch,edge_attr = None,edge_weight=None, return_both_rep=False)
        logits = self.cls_header(g) #! (B,3)
        return logits

    def train_labelled_one_step(self,data):
        # self.encoder_model.train()
        data = data.to(self.device)
        y = data.y
        g = self.gnn(data.x, data.edge_index, batch=data.batch,edge_attr = data.edge_attr,edge_weight=None, return_both_rep=False)
        logits = self.cls_header(g)
        loss = self.ce_loss(logits, y)
        # edge_reg = (torch.sum(edge_weight)/tot_edges-self.edge_budget)**2
        return loss


    def fit(self,dataloader,valid_dloader,test_dloader=None,epochs=50):
        for e in range(epochs):
            print (colored(f'Current Epoch {e}','red','on_yellow'))
            if self.stop_training:
                break
            erm_losses = 0.
            total_losses = 0.
            steps = 0
            for data in dataloader:
                data = data.to(self.device)
                self.optimizer.zero_grad()
                labelled_loss = self.train_labelled_one_step(data)

                labelled_loss.backward()
                self.optimizer.step()
                erm_losses += labelled_loss.item()
                steps +=1
                # use colored to print three losses
            
            print (colored(f'Epoch {e}: Labelled Loss: {erm_losses/steps}','red','on_white'))
            train_metric_score = self.evaluate_model(dataloader,'train',is_dataloader=True)
            val_metric_score = self.evaluate_model(valid_dloader,'valid',is_dataloader=True)
            self.train_metrics.append(train_metric_score)
            self.val_metrics.append(val_metric_score)
            # self.train_grad_sim.append(train_avg_grad_sim)
            # self.val_grad_sim.append(val_avg_grad_sim)     
            if test_dloader is not None:
                test_metric_score= self.evaluate_model(test_dloader,'test',is_dataloader=True)
                self.test_metrics.append(test_metric_score)
                # self.test_grad_sim.append(test_avg_grad_sim)
            self.valid_metric_list.append((val_metric_score,test_metric_score))
            self.train()
        

    def evaluate_model(self, data_input,phase,is_dataloader=True,best_edge_weight=None):
        """
        For cls header evaluation
        Evaluate the model on a given dataset.

        Args:
        - dataloader: DataLoader for the dataset to evaluate.
        - phase: Phase of evaluation ('valid' or 'test') to control output messaging.

        Returns:
        - The average metric value across the dataset.
        - The average cosine similarity between SSL and ERM gradients.
        """

        self.eval()  # Set model to evaluation mode
        logits_list = []
        labels_list = []
        steps = 0
        with torch.no_grad():
            if not is_dataloader:
                data_input = [data_input]
            for data in data_input:
                data = data.to(self.device)
                # Metric computation
                if best_edge_weight is not None:
                    edge_weight = best_edge_weight
                else:
                    edge_weight = None
                

                g = self.gnn(data.x, data.edge_index, batch=data.batch,edge_attr = data.edge_attr,edge_weight=edge_weight, return_both_rep=False)
                logits = self.cls_header(g)
                if not is_dataloader:
                    self.train()
                    return logits
                
                logits_list.append(logits)
                labels_list.append(data.y.view(-1,))
                steps += 1

            all_logits = torch.cat(logits_list, dim=0)
            all_labels = torch.cat(labels_list, dim=0)
            
            # Compute metric with all logits and labels
            if self.metric_name=='acc':
                metric_score = self.metric_func(all_logits, all_labels).item()
            
            if self.metric_name=='auc':
                metric_score = self.metric_func(all_logits[:,1], all_labels).item() # use pos logits
            
            
            if phase.lower()=='valid':
                if metric_score>=self.best_valid_metric:
                    self.best_valid_metric = metric_score
                    self.epochs_since_improvement=0
                    # save model
                    self.best_states = deepcopy(self.state_dict())
                else:
                    self.epochs_since_improvement+=1
                    if self.epochs_since_improvement>=self.early_stop_epochs:
                        self.stop_training=True
                        print (colored(f'Early Stopping: No improvement for {self.early_stop_epochs} epochs','red','on_white'))
            if phase=='test':
                print(colored(f'{phase} Phase: Average {self.metric_name}: {metric_score}', 'blue','on_yellow'))
            else:
                print(colored(f'{phase} Phase: Average {self.metric_name}: {metric_score}', 'blue','on_white'))
            self.train()
            return metric_score
    
    