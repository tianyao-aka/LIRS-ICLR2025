import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import sys
from tqdm import tqdm

import os.path as osp

from torch.optim import Adam

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

from torch_geometric.explain import Explainer, Explanation, GNNExplainer,PGExplainer
from torch_geometric.explain.config import ExplainerConfig, ModelMode

from torch_geometric.explain.metric import groundtruth_metrics,fidelity
from torchmetrics import AUROC

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


class ModelExplainer(BaseModel):
    def __init__(self, model_to_explain, lr=2e-3,epochs = 20, topK=5, device='cpu',metric_name='auc', train_loader=None,entire_train_loader=None,entire_train_loader_batch=None, **args):

        super(ModelExplainer, self).__init__()
        self.device = device
        self.model = model_to_explain
        self.model.eval()
        self.model.to(device)
        assert device is not None, "Please specify 'device'!"
        self.explainer = Explainer(
        model=self.model,
        algorithm=PGExplainer(epochs=epochs, lr=0.002),
        # PGExplainer only supports a phenomenon explanation type which means it only
        # generates explanations over the expected output, not the predicted output.
        explanation_type='phenomenon',
        # PGExplainer does not support node masks! This means that we cannot
        # generate the node features bar chart we were able to for the previous
        # explainers.
        edge_mask_type='object',
        model_config=dict(mode='multiclass_classification', task_level='graph', return_type='raw'),
        threshold_config=dict(threshold_type='topk', value=topK)
    )
        self.tr_loader = train_loader
        self.entire_train_loader = entire_train_loader
        self.entire_train_loader_batch = entire_train_loader_batch
        self.epochs = epochs
        self.lr = lr
        self.topK = topK
        self.metric = metric_name
        self.node_list = []
        
        
        self.auc_func = AUROC(task='binary').to(device)
        self.explainer.algorithm.mlp.to(device)

        
    def train_explainer(self):
        for epoch in range(self.epochs):
            print ('epoch:',epoch)
            for batch in self.tr_loader:
                batch = batch.to(self.device)
                loss = self.explainer.algorithm.train(epoch, self.model, batch.x, batch.edge_index,
                                                target=batch.y,batch=batch.batch)
    
    def get_topK_nodes(self,edge_mask,edge_index):
        
        """
        Retrieves the unique nodes connected by the top K non-zero edges with the highest scores in the edge_mask.
        
        Args:
        edge_mask (torch.Tensor): A tensor containing the scores of edges.
        edge_index (torch.Tensor): A 2xN tensor where each column represents an edge as [source_node, target_node].
        topK (int): The number of top edges to consider, but only consider non-zero edges.
        
        Returns:
        torch.Tensor: A tensor containing the unique node indices involved in the top K edges.
        """
        # Filter out zero-valued edges and get corresponding indices
        topK = self.topK
        nonzero_mask_indices = edge_mask.nonzero(as_tuple=True)[0]
        nonzero_edge_mask = edge_mask[nonzero_mask_indices]

        # Determine the actual number of top edges to consider based on the number of non-zero edges
        actual_topK = min(topK, len(nonzero_edge_mask))

        # Get indices of the top K highest values among the non-zero edge scores
        _, topk_indices = torch.topk(nonzero_edge_mask, actual_topK)
        
        # Convert local topK indices back to original indices in edge_index
        topk_original_indices = nonzero_mask_indices[topk_indices]
        # Use the indices to fetch the corresponding nodes from edge_index
        topk_edges = edge_index[:, topk_original_indices]

        # Retrieve the unique nodes from the selected edges
        unique_nodes = torch.unique(topk_edges)
        return unique_nodes

    def eval_auc(self):
        preds = []
        targets = []
        for batch in self.entire_train_loader:
            batch = batch.to(self.device)
            explanation = self.explainer(batch.x,batch.edge_index,target=batch.y,batch=batch.batch)
            num_nodes = batch.x.shape[0]
            node_tensor = torch.zeros(num_nodes)
            node_explained = self.get_topK_nodes(explanation.edge_mask, batch.edge_index)
            self.node_list.append(node_explained)
            node_tensor.index_put_((torch.tensor(node_explained),), torch.tensor(1.0))
            preds.append(node_tensor)
            targets.append(batch.node_label)

        preds = torch.cat(preds,dim=0)
        targets = torch.cat(targets,dim=0)    
        score = self.auc_func(preds.to(self.device),targets.long().to(self.device))
        return score
    
    def eval_fidelity(self):
        for batch in self.entire_train_loader:
            batch = batch.to(self.device)
            explanation = self.explainer(batch.x,batch.edge_index,target=batch.y,batch=batch.batch)
            node_explained = self.get_topK_nodes(explanation.edge_mask, batch.edge_index)
            self.node_list.append(node_explained)
        
        pos_fi = []
        for batch in self.entire_train_loader_batch:
            batch = batch.to(self.device)
            explanation = self.explainer(batch.x,batch.edge_index,target=batch.y,batch=batch.batch)
            pos_fidelity,_ = fidelity(self.explainer,explanation)
            pos_fi.append(pos_fidelity)
        return np.mean(pos_fi)

    def run(self):
        self.train_explainer()
        if self.metric != 'auc':
            score = self.eval_fidelity()
        else:
            score = self.eval_auc()
        return score
        
    
    def write_node_list(self,fpath,name):
        os.makedirs(os.path.dirname(fpath), exist_ok=True)
        fpath = os.path.join(fpath,name)
        torch.save(self.node_list, fpath)
        
        
    