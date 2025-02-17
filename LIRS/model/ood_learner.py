import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from model.gin import GIN
from model.vgin import VGIN
# from torch_geometric.nn import GIN, GCN
from torchmetrics import Accuracy, AUROC
from model.HSIC import HSIC

from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

import torch.optim.lr_scheduler as lr_scheduler

from torchmetrics import AUROC,Accuracy
import numpy as np
import pandas as pd
import random
import string
from termcolor import colored
from copy import deepcopy
import sys

def track_cluster_labels(loader):
    """
    Tracks the number of cluster labels for each label in batch.y.

    Args:
        loader (DataLoader): The data loader containing batches of data.

    Returns:
        None
    """
    # Initialize the cluster count dictionary with predefined labels and cluster IDs
    cluster_count = {
        0: {0: 0, 1: 0, 2: 0},
        1: {0: 0, 1: 0, 2: 0},
        2: {0: 0, 1: 0, 2: 0}
    }

    # Iterate over the batches in the loader
    cnt = 0
    while 1:
        cnt +=1
        batch = next(iter(loader))
        # Iterate over the samples in the batch
        for label, cluster_id in zip(batch.y, batch.cluster_id):
            # Convert to int for counting purposes
            label = int(label)
            cluster_id = int(cluster_id)
            # Increment the count for the corresponding label and cluster_id
            cluster_count[label][cluster_id] += 1
        if cnt>140: break

    # Print the cluster counts for each label in y
    for label, clusters in cluster_count.items():
        print(f"Label {label}:")
        for cluster_id, count in clusters.items():
            print(f"  Cluster {cluster_id}: {count} samples")



class ModelTrainer(nn.Module):
    def __init__(self, nfeat, nhid, nclass, nlayers=2, dropout=0.5, edge_dim=-1, save_mem=True, jk='last', 
                 node_cls=False, pooling='sum', with_bn=False, weight_decay=5e-6, lr=1e-3,
                 patience=50, early_stop_epochs=10, hsic_penalty=0.1, project_layer_num=2,
                 base_gnn='gin', kernel_method='rbf', sigma=0.1, valid_metric='acc', device='cpu', **args):
        super(ModelTrainer, self).__init__()
        
        self.hsic = HSIC(kernel_method, sigma,device=device)
        self.hsic.to(device)
        self.delta_acc_arr = None
        self.valid_test_delta_acc_list = []
        dataset_name = args.get('dataset')
        n_clusters = args.get('num_clusters',0)
        mol_encoder = args.get('mol_encoder',False)
        self.use_lr_scheduler = args.get('use_lr_scheduler',False)
        vGIN = args.get('vGIN',False)
        assert n_clusters>0, "Number of clusters should be greater than 0"
        
        if base_gnn == 'gin':
            if not vGIN:
                self.gnn = GIN(nfeat, nhid, nclass, nlayers, dropout=dropout, edge_dim=edge_dim, jk=jk, node_cls=node_cls, 
                            pooling=pooling, with_bn=with_bn, weight_decay=weight_decay,dataset = dataset_name,mol_encoder=mol_encoder)
            else:
                self.gnn = VGIN(nfeat, nhid, nclass, nlayers, dropout=dropout, edge_dim=edge_dim, jk=jk, node_cls=node_cls, 
                            pooling=pooling, with_bn=with_bn, weight_decay=weight_decay,dataset_name=dataset_name,mol_encoder=mol_encoder)
            self.gnn.to(device)
        elif base_gnn == 'gcn':
            self.gnn = GCN(nfeat, nhid, nclass, nlayers=nlayers, dropout=dropout, save_mem=save_mem, jk=jk, 
                           node_cls=node_cls, pooling=pooling, with_bn=with_bn, weight_decay=weight_decay)
            self.gnn.to(device)
        self.cls_header = self.create_mlp(nhid, nhid, nclass, project_layer_num).to(device)
        self.cluster_cls_header = self.create_mlp(nhid, nhid, n_clusters, project_layer_num).to(device)
        self.cluster_ce_loss = nn.CrossEntropyLoss(reduction='none').to(device)
        
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = device
        self.penalty = hsic_penalty
        self.optimizer = Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.ce_loss = nn.CrossEntropyLoss().to(device)
        self.metric_func = (Accuracy(task='multiclass', num_classes=nclass, top_k=1).to(device) 
                            if valid_metric == 'acc' else AUROC(task='binary').to(device))
        self.metric_name = valid_metric
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.1)
        self.min_lr = 1e-4
                
        self.balance_sampling = args.get("balance_sampler",False)
        self.intra_class_clustering = args.get("intra_cluster_labels",False)
        self.intra_class_penalty = args.get("intra_cluster_penalty",0.)
        if self.intra_class_clustering:
            pass
        
        print (colored(f'use balance sampling or not: {self.balance_sampling}','red'))
        print (colored(f'use intra class clustering or not: {self.intra_class_clustering}, penalty is:{self.intra_class_penalty}','red'))
        
        self.train_metrics = []
        self.val_metrics = []
        self.test_metrics = []
        self.valid_metric_list = []
        self.meta_valid_metric_list = []
        self.best_valid_metric = -1.
        self.test_metric = -1.
        self.early_stop_epochs = early_stop_epochs
        self.epochs_since_improvement = 0
        self.stop_training = False

    
    def create_mlp(self, input_dim, hidden_dim, output_dim, num_layers, cls=True):
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        if cls:
            layers.append(nn.Linear(hidden_dim, output_dim))
        return nn.Sequential(*layers)
    
    def parameters(self):
        return list(self.gnn.parameters()) + list(self.cls_header.parameters())


    def train_single_step(self,data,uniform_sampling=False):
        # self.encoder_model.train()
        hsic_loss = 0.
        intra_cluster_loss = 0.
        data = data.to(self.device)
        y = data.y
        
        size = len(y)
        g = self.gnn(data.x, data.edge_index, batch=data.batch,edge_attr = data.edge_attr,return_both_rep=False)
        logits = self.cls_header(g)
        loss = self.ce_loss(logits, y.view(-1,).long())
        if self.penalty>0:
            assert data.spu_rep.shape[1]==g.shape[1], "Shape mismatch of graph_emb and spu_emb!"
            hsic_loss = self.hsic.compute_hsic(g,data.spu_rep)
        if self.intra_class_penalty>0:
            cluster_labels = data.intra_cluster_labels
            target_logits = data.intra_cluster_pred_logits  
            sample_weights = data.sample_weights
            pred_logits = self.cluster_cls_header(g)  # (N,n_clusters)
            intra_cluster_loss = self.cross_entropy_with_soft_targets(pred_logits,target_logits,sample_weights)
            
            # loss_per_sample = self.cluster_ce_loss(logits,data.binary_cluster_id.long())
            # pos_mask = data.binary_cluster_id==1
            # neg_mask = data.binary_cluster_id==0
            # if uniform_sampling:
            #     intra_cluster_loss = (torch.sum(loss_per_sample[pos_mask])+torch.sum(0.5*loss_per_sample[neg_mask]))/size
            # else:
            #     sample_reweight_ratio = data.binary_cluster_count[:,0]/(data.binary_cluster_count[:,1]) # num_samples_in_big_cluster/num_samples_in_small_cluster
            #     intra_cluster_loss = (torch.sum(loss_per_sample[pos_mask]) + torch.sum(loss_per_sample[neg_mask]*sample_reweight_ratio[neg_mask]))/size
        
        return loss,hsic_loss,intra_cluster_loss

    def fit(self,train_loader,valid_dloader,test_dloader=None,epochs=100):
        for e in range(epochs):
            current_lr = self.scheduler.get_last_lr()[0]
            if e<=1:
                self.epochs_since_improvement=0
                self.stop_training=False
            
            if self.use_lr_scheduler and current_lr>=self.min_lr:
                self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0]
                print(f"Epoch {e+1}, Learning Rate: {current_lr}")
            
            print (colored(f'Current Epoch {e}','red','on_yellow'))
            # if e==10 and self.pe>0:
            #     self.useAutoAug,self.encoder_model.learnable_aug,self.epochs_since_improvement,self.edge_uniform_penalty,self.penalty,self.edge_penalty = state
            #     self.best_valid_metric = 0.
            if self.stop_training:
                break
            total_losses = 0.
            total_hsic_loss = 0.
            total_intra_cluster_loss = 0.
            steps = 0
            if not self.balance_sampling:
                for data in train_loader:
                    data = data.to(self.device)
                    self.optimizer.zero_grad()
                    
                    erm_loss, hsic_loss, intra_cluster_loss = self.train_single_step(data,uniform_sampling=self.balance_sampling)
                    # print (colored(f'ERM Loss: {erm_loss.item()} Reg Loss: {reg_loss.item()}','red','on_white'))
                    loss = erm_loss + self.penalty*hsic_loss + self.intra_class_penalty*intra_cluster_loss
                    loss.backward()
                    self.optimizer.step()
                    total_losses += loss.item()
                    total_hsic_loss += hsic_loss.item() if torch.is_tensor(hsic_loss) else hsic_loss
                    total_intra_cluster_loss+= intra_cluster_loss.item() if torch.is_tensor(intra_cluster_loss) else intra_cluster_loss
                    steps +=1
            else:
                N = len(train_loader)
                cnt = 0
                while True:
                    if cnt>=N+20: break
                    data = next(iter(train_loader))
                    data = data.to(self.device)
                    self.optimizer.zero_grad()
                    erm_loss, hsic_loss, intra_cluster_loss = self.train_single_step(data,uniform_sampling=self.balance_sampling)
                    # print (colored(f'ERM Loss: {erm_loss.item()} Reg Loss: {reg_loss.item()}','red','on_white'))
                    
                    loss = erm_loss + self.penalty*hsic_loss + self.intra_class_penalty*intra_cluster_loss
                    loss.backward()
                    self.optimizer.step()
                    total_losses += loss.item()
                    total_hsic_loss += hsic_loss.item() if torch.is_tensor(hsic_loss) else hsic_loss
                    total_intra_cluster_loss+= intra_cluster_loss.item() if torch.is_tensor(intra_cluster_loss) else intra_cluster_loss
                    steps +=1
                    cnt +=1
            
            print (colored(f'Epoch {e} total Loss: {total_losses/steps}, hsic loss: {total_hsic_loss/steps}, intraCluster loss:{total_intra_cluster_loss/steps}','red','on_white'))
            
            #! run time calc, no need for this
            # train_metric_score = self.evaluate_model(train_loader,'train')
            val_metric_score = self.evaluate_model(valid_dloader,'valid')
            # self.train_metrics.append(train_metric_score)
            # self.val_metrics.append(val_metric_score)
            
            # if test_dloader is not None:
            #     test_metric_score= self.evaluate_model(test_dloader,'test')
            #     self.test_metrics.append(test_metric_score)
            #     if self.delta_acc_arr is not None:
            #         self.valid_test_delta_acc_list.append((val_metric_score,self.delta_acc_arr))
                
                
            # self.valid_metric_list.append((val_metric_score,test_metric_score))
            # # print metrics in all stages
            # print (colored(f'Epoch: {e}: Train Metric: {train_metric_score} Val Metric: {val_metric_score} Test Metric: {test_metric_score}','red','on_white'))


    def evaluate_model(self, data_loader, phase):
            self.eval()
            logits_list = []
            labels_list = []
            logits_list_after_removal = []
            with torch.no_grad():
                for data in data_loader:
                    data = data.to(self.device)
                    g = self.gnn(data.x, data.edge_index, batch=data.batch, edge_attr=data.edge_attr, return_both_rep=False)
                    logits = self.cls_header(g)
                    logits_list.append(logits)
                    labels_list.append(data.y)
            
            all_logits = torch.cat(logits_list, dim=0)
            all_labels = torch.cat(labels_list, dim=0)
            
            if self.metric_name == 'acc':
                metric_score = self.metric_func(all_logits, all_labels).item()
            elif self.metric_name == 'auc':
                metric_score = self.metric_func(all_logits[:, 1], all_labels).item()

            if 'node_label' in data and phase=='test':
                # run Gc removal and calc delta_acc
                # print ('calc delta acc')
                with torch.no_grad():
                    for data in data_loader:
                        data = data.to(self.device)
                        edge_index,x,_ = self.remove_one_Gc(data.x, data.edge_index, data.node_label, data.ptr)
                        data.edge_index = edge_index
                        data.x = x
                        data = data.to(self.device)
                        g = self.gnn(data.x, data.edge_index, batch=data.batch, edge_attr=data.edge_attr, return_both_rep=False)
                        logits = self.cls_header(g)
                        logits_list_after_removal.append(logits)
                all_logits_after_removal = torch.cat(logits_list_after_removal, dim=0)
                self.delta_acc_arr = self.compute_difference(all_logits, all_logits_after_removal, all_labels).detach().cpu().numpy()


            if phase.lower() == 'valid':
                print ('valid,',self.best_valid_metric,metric_score )
                if metric_score > self.best_valid_metric:
                    self.best_valid_metric = metric_score
                    self.epochs_since_improvement = 0
                    self.best_states = deepcopy(self.state_dict())
                else:
                    self.epochs_since_improvement += 1
                    if self.epochs_since_improvement >= self.early_stop_epochs:
                        self.stop_training = True
            self.train()
            return metric_score


    def DFR(self, valid_loader, test_loader, method="mlp"):
        """
        Performs Deep Feature Reweighting (DFR) using the learned GNN embeddings. 
        The embeddings are used to train a downstream classifier (MLP or Logistic Regression) 
        on the validation set and evaluated on the test set.

        Args:
            valid_loader (DataLoader): DataLoader for the validation dataset.
            test_loader (DataLoader): DataLoader for the test dataset.
            method (str): The method to use for downstream classifier. Either 'mlp' or 'linear'.

        Returns:
            Tuple[float, float]: Accuracy scores on the validation set and test set.
        """

        # Collect embeddings for validation set
        valid_embeddings = []
        valid_labels = []
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            for data in valid_loader:
                data = data.to(self.device)
                gnn_output = self.gnn(data.x, data.edge_index, batch=data.batch, edge_attr=data.edge_attr, return_both_rep=False)
                valid_embeddings.append(gnn_output.cpu())
                valid_labels.append(data.y.cpu())

        valid_embeddings = torch.cat(valid_embeddings, dim=0)
        valid_labels = torch.cat(valid_labels, dim=0)

        # Collect embeddings for test set
        test_embeddings = []
        test_labels = []
        with torch.no_grad():
            for data in test_loader:
                data = data.to(self.device)
                gnn_output = self.gnn(data.x, data.edge_index, batch=data.batch, edge_attr=data.edge_attr, return_both_rep=False)
                test_embeddings.append(gnn_output.cpu())
                test_labels.append(data.y.cpu())

        test_embeddings = torch.cat(test_embeddings, dim=0)
        test_labels = torch.cat(test_labels, dim=0)

        # Train a downstream classifier on the validation embeddings
        if method == "mlp":
            clf = MLPClassifier(hidden_layer_sizes=(valid_embeddings.size(1), valid_embeddings.size(1) // 2), max_iter=500, random_state=42)
        
        elif method == "linear":
            # Apply L2 (Euclidean) normalization to the embeddings before Logistic Regression
            scaler = StandardScaler()
            valid_embeddings = scaler.fit_transform(valid_embeddings.numpy())  # Normalize valid embeddings
            test_embeddings = scaler.transform(test_embeddings.numpy())  # Normalize test embeddings
            
            clf = LogisticRegression(max_iter=1000, random_state=42)
        else:
            raise ValueError(f"Unknown method: {method}")

        clf.fit(valid_embeddings, valid_labels.numpy())

        # Evaluate on validation and test sets
        valid_predictions = clf.predict(valid_embeddings)
        valid_acc = accuracy_score(valid_labels.numpy(), valid_predictions)

        test_predictions = clf.predict(test_embeddings)
        test_acc = accuracy_score(test_labels.numpy(), test_predictions)

        return valid_acc, test_acc


    def remove_one_Gc(self,X,edge_index, node_label, ptr,num_Gc=2):
        """
        Processes the input graphs to delete edges and set node features to zero based on node labels.
        
        Parameters:
        edge_index (torch.Tensor): The edge index tensor of shape (2, E).
        node_label (torch.Tensor): The node label tensor of shape (N,).
        X (torch.Tensor): The node feature matrix of shape (N, D).
        ptr (torch.Tensor): The pointer tensor that holds indices for each graph in the batch, shape (batch_size+1,).
        
        Returns:
        torch.Tensor: Updated edge_index tensor.
        torch.Tensor: Updated node feature matrix X.
        torch.Tensor: Updated node label tensor.
        """
        
        batch_size = ptr.size(0) - 1
        new_edge_index_list = []
        new_X = X.clone()
        new_node_label = node_label.clone()

        for i in range(batch_size):
            start, end = ptr[i], ptr[i + 1]
            node_indices = torch.arange(start, end)
            node_labels = node_label[start:end]
            
            # Calculate K
            count_ones = (node_labels == 1).sum().item()
            K = count_ones // num_Gc
            K = K*(num_Gc-1)
            
            if K > 0:
                one_indices = (node_labels == 1).nonzero(as_tuple=False).view(-1) + start
                last_k_indices = one_indices[-K:]
                # Setting the last K node labels to zero
                new_node_label[last_k_indices] = 0
                
                # Deleting edges
                edge_mask = torch.ones(edge_index.size(1), dtype=torch.bool).to(self.device)
                
                for idx in last_k_indices:
                    edge_mask &= (edge_index[0] != idx) & (edge_index[1] != idx)
                edge_index = edge_index[:, edge_mask]
                # Setting node features to zero
                new_X[last_k_indices] = 0
                
        return edge_index, new_X, new_node_label
    

    def compute_difference(self,logits, logits2, labels):
        """
        Computes the difference between corresponding rows and ground truth class columns of logits and logits2
        after normalizing using softmax.
        
        Parameters:
        logits (torch.Tensor): Tensor of shape (N, C).
        logits2 (torch.Tensor): Tensor of shape (N, C).
        labels (torch.Tensor): Tensor of shape (N,).
        
        Returns:
        torch.Tensor: The difference L1 - L2.
        """
        # Step 1: Normalize logits and logits2 using softmax
        logits = F.softmax(logits, dim=1)
        logits2 = F.softmax(logits2, dim=1)
        
        # Step 2: Get the correct predicted sample indices
        predicted = torch.argmax(logits, dim=1)
        correct_indices = (predicted == labels).nonzero(as_tuple=False).view(-1)
        
        # Step 3: Get the rows in logits and logits2, and corresponding columns for the ground truth classes
        L1 = logits[correct_indices, labels[correct_indices]]
        L2 = logits2[correct_indices, labels[correct_indices]]
        # Step 4: Return L1 - L2
        return L1 - L2


    def cross_entropy_with_soft_targets(self,logits, targets,sample_weights):
        """
        Calculates the cross entropy loss with soft targets manually.

        Parameters:
        logits (torch.Tensor): Logits of shape (N, C), where C is the number of classes.
        targets (torch.Tensor): Soft targets of shape (N, C), where C is the number of classes.
        targets (torch.Tensor): label targets of shape (N, )
        Returns:
        torch.Tensor: Cross entropy loss 
        """

        log_probs = F.log_softmax(logits, dim=1)
        loss = -torch.sum(targets * log_probs, dim=1)  # Sum over the class dimension
        return torch.mean(loss*sample_weights)
    
        