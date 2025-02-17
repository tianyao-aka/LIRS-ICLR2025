import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch_geometric.utils import to_undirected,add_self_loops,remove_self_loops
from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.data import Data,DataLoader
from torch_geometric.datasets import TUDataset
from torch_geometric import transforms as T
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import degree
from torch_scatter import scatter_add
from torch_geometric.utils import degree
from sklearn.model_selection import StratifiedKFold, KFold
from time import time
import random
from collections import defaultdict
import os
from torch_geometric.data import Data, Dataset

from utils.functionalUtils import tensor_difference
from torch_geometric.data import Batch
import random

def k_fold(dataset, folds):
    #! for TU Datasets
    skf = StratifiedKFold(folds, shuffle=True, random_state=1)

    test_indices, train_indices = [], []
    for _, idx in skf.split(torch.zeros(len(dataset)), dataset.data.y[dataset.indices()]):
        test_indices.append(torch.from_numpy(idx))

    val_indices = [test_indices[i - 1] for i in range(folds)]

    for i in range(folds):
        train_mask = torch.ones(len(dataset), dtype=torch.uint8)
        train_mask[test_indices[i]] = 0
        train_mask[val_indices[i]] = 0
        train_indices.append(train_mask.nonzero().view(-1))

    return train_indices, test_indices, val_indices

class DegreeTransform(object):
    # combine position and intensity feature, ignore edge value
    def __init__(self) -> None:
        self.deg_func = T.OneHotDegree(max_degree=10000)

    def __call__(self, data):
        data = self.deg_func(data)
        N = data.x.shape[0]
        degrees = degree(data.edge_index[0],num_nodes=N).view(-1,1).float()
        max_degree = degrees.max().item()
        degrees = degrees/max_degree
        val = torch.cat([degrees,data.x],dim=1)
        val = val[:,:65]
        data.x = val
        return data



def generate_tensor_lists(K, T):
    """
    Generates two lists of tensors based on the specified parameters.

    Args:
    K (int): The upper bound (exclusive) for the range of index tensors in list A.
    T (int): The number of trials, determining the length of the lists.

    Returns:
    tuple: A tuple containing two lists of tensors, A and B.
           A - list of tensors, each tensor is an index tensor [0, 1, 2, ..., K-1].
           B - list of tensors, each tensor is an index tensor [0].
    """
    A = [torch.arange(K).long() for _ in range(T)]
    B = [torch.tensor([0]).long() for _ in range(T)]

    return A, B

def generate_balanced_subsets_and_complements(L, K, dataset, C, num_trials=20, seed=1):
    """
    Generate balanced deterministic subsets of size K and their complements from a given list L.
    
    Parameters:
    - L (list): The original list of indices.
    - K (int): The size of labelled subset
    - dataset (PyG Dataset): The PyTorch Geometric dataset containing the labels.
    - C (int): The number of classes.
    - num_trials (int): The number of subsets to generate. Default is 20.
    - seed (int): The random seed for deterministic behavior. Default is 1.
    
    Returns:
    - subsets (list of tensors): A list containing the generated subsets as tensors.
    - complements (list of tensors): A list containing the complement sets of the generated subsets as tensors.
    """
    
    # Set the random seed for deterministic behavior
    random.seed(seed)
    
    # Initialize the lists to store the subsets and their complements
    subsets = []
    complements = []
    
    # Separate indices by class label
    class_indices = defaultdict(list)
    for i in L:
        label = dataset[i].y.item()  # Assuming label is a tensor of shape [1]
        class_indices[label].append(i)
    
    # Validate that each class has enough samples
    for label, indices in class_indices.items():
        if len(indices) < K // C:
            raise ValueError(f"Class {label} has insufficient samples.")
    
    # Convert the original list L to a set for faster set operations
    original_set = set(L)
    
    # Generate balanced subsets and their complements
    for _ in range(num_trials):
        subset = []
        for label, indices in class_indices.items():
            subset += random.sample(indices, K // C)
        
        subset = torch.tensor(subset)  # Convert list to tensor
        complement = torch.tensor(list(original_set - set(subset.tolist())))  # Convert list to tensor
        
        subsets.append(subset)
        complements.append(complement)
        
    return subsets, complements

class KHopTransform(BaseTransform):
    def __init__(self, K=[1, 2, 3],agg='sum'):
        super(KHopTransform, self).__init__()
        self.K = K
        self.agg=agg

    def __call__(self, data):
        device = data.edge_index.device
        N = data.num_nodes
        edge_index = data.edge_index
        
        hop_features = []
        for k in self.K:
            # Create adjacency matrix A and raise it to the power of k
            A_k = torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.size(1)), size=(N, N))
            A_k = torch.sparse.mm(A_k, A_k) if k == 2 else A_k  # if k > 2, need to multiply more times
            
            D_k = A_k.to_dense().sum(dim=1)
            D_k_inv = torch.diag(D_k.pow(-1))
            D_k_inv = torch.where(torch.isinf(D_k_inv), torch.tensor(0.0), D_k_inv)
            hop_feature = torch.mm(A_k.to_dense(), D_k_inv @ data.x.float())
            hop_features.append(hop_feature)

        data.hop1_feature = hop_features[0]
        if 2 in self.K:
            data.hop2_feature = hop_features[1]
        if 3 in self.K:
            data.hop3_feature = hop_features[2]
        if "edge_attr" in data and self.agg=='sum':
                edge_feature = scatter_add(data.edge_attr, data.edge_index[1], dim=0, dim_size=data.num_nodes)
                data.edge_features = edge_feature
        return data
    
class IDTransform(BaseTransform):
    def __init__(self,agg='sum'):
        super(IDTransform, self).__init__()
        self.id = 0
    def __call__(self, data):
        data.id = self.id
        self.id += 1
        return data

def load_graph_embeddings(base_path="experiment_results/SSL_embedding/MUTAG/MVGRL/",embName = 'graph_embedding.pt',str_match=None):
    """
    Load graph embeddings from multiple folders within a given base path.
    
    Parameters:
    - base_path (str): The base directory where the folders are located.
    
    Returns:
    - embeddings_dict (dict): A dictionary where keys are folder names and values are loaded tensors.
    """
    
    # Initialize an empty dictionary to store the loaded tensors
    embeddings_dict = {}
    
    # Loop through all folders in the base directory
    for folder_name in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder_name)
        
        # Check if the path is a directory
        if os.path.isdir(folder_path):
            # print ("Loading embeddings from folder:", folder_name)
            # Construct the full path to the graph_embedding.pt file
            file_path = os.path.join(folder_path, embName)
            
            # Check if the file exists
            if os.path.exists(file_path):
                if str_match is None:
                    # Load the tensor using PyTorch
                    tensor = torch.load(file_path)
                    
                    # Add the tensor to the dictionary
                    embeddings_dict[folder_name] = tensor
                else:
                    if str_match in folder_name:
                        # Load the tensor using PyTorch
                        tensor = torch.load(file_path)
                        
                        # Add the tensor to the dictionary
                        embeddings_dict[folder_name] = tensor
                
    return embeddings_dict


class CustomDataset(Dataset):
    def __init__(self, dataset1, dataset2):
        super(CustomDataset, self).__init__()
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.length1 = len(dataset1)
        self.length2 = len(dataset2)

    def len(self):
        return self.length1 + self.length2

    def get(self, idx):
        if idx < self.length1:
            data = self.dataset1[idx]
            data.is_noisy = torch.tensor([False])
            data.id = idx
        else:
            data = self.dataset2[idx - self.length1]
            data.is_noisy = torch.tensor([True])
        return data


class CustomKD_Dataset(Dataset):
    def __init__(self, dataset1, dataset2, dataset3, svm_proba1, svm_proba2, svm_proba3):
        """
        datasetX: pyG dataset with different partitions using indices
        svm_probaX: pytorch tensors of shape (N,5,C), assuming 5 clfs used.
        """
        
        super(CustomKD_Dataset, self).__init__()
        self.dataset1 = dataset1
        self.dataset2 = dataset2 if dataset2 is not None else []
        self.dataset3 = dataset3
        self.svm_proba1 = svm_proba1
        self.svm_proba2 = svm_proba2
        self.svm_proba3 = svm_proba3
        self.length1 = len(dataset1)
        self.length2 = len(self.dataset2)
        self.length3 = len(dataset3)

    def len(self):
        return self.length1 + self.length2 + self.length3

    def get(self, idx):
        if idx < self.length1:
            data = self.dataset1[idx]
            data.is_noisy = torch.tensor([0])
            data.svm_proba = self.svm_proba1[idx]
        elif idx < self.length1 + self.length2:
            data = self.dataset2[idx - self.length1]
            data.is_noisy = torch.tensor([1])
            data.svm_proba = self.svm_proba2[idx - self.length1]
        else:
            data = self.dataset3[idx - self.length1 - self.length2]
            data.is_noisy = torch.tensor([2])  # Using 2 to distinguish from dataset1 and dataset2
            data.svm_proba = self.svm_proba3[idx - self.length1 - self.length2]
        data.id = idx
        return data
    
class FilteredDataset(InMemoryDataset):
    def __init__(self, root, original_dataset, filter_label, transform=None, pre_transform=None):
        super(FilteredDataset, self).__init__(root, transform, pre_transform)
        self.kept_indices = []
        self.filtered_data_list = []

        # Determine the unique labels excluding the filter_label
        unique_labels = set()
        for data in original_dataset:
            if data.y.item() != filter_label:
                unique_labels.add(data.y.item())

        # Create a mapping from original labels to new labels starting from 0
        label_mapping = {label: i for i, label in enumerate(sorted(unique_labels))}
        if 'id' in data:
            # Create a mapping for re-indexing IDs
            id_mapping = {}
            new_id = 0
            for data in original_dataset:
                if data.y.item() != filter_label:
                    id_mapping[data.id] = new_id
                    new_id += 1

        # Filter the original dataset, apply the label mapping, and apply the ID mapping
        for i, data in enumerate(original_dataset):
            if data.y.item() != filter_label:
                # Apply the label mapping
                data.y = torch.tensor([label_mapping[data.y.item()]], dtype=data.y.dtype)
                # Apply the ID mapping
                if 'id' in data:
                    data.id = id_mapping[data.id]

                self.filtered_data_list.append(data)
                self.kept_indices.append(i)

        self.data, self.slices = self.collate(self.filtered_data_list)

    
    def get_kept_indices(self):
        """
        Returns the indices of the data points that were kept after filtering.

        Returns:
        list: The list of indices that were kept.
        """
        return self.kept_indices
    
    def _download(self):
        pass

    def _process(self):
        pass
    
    
class ModifiedDataset(Dataset):  # add multiplr attr into datasets
    def __init__(self, original_dataset, attr_name, attr_value):
        # Initialize the parent class
        super(ModifiedDataset, self).__init__()
        self.dataset = original_dataset
        self.attr_name = attr_name
        self.attr_value = attr_value
        self.N = len(attr_name)

    def len(self):
        # Return the length of the dataset
        return len(self.dataset)

    def get(self, idx):
        # Get an item, modify it, and return it
        data = self.dataset.get(idx)
        setattr(data, self.attr_name, self.attr_value[idx].unsqueeze(0))
        return data


def process_batch(batch, C, num_samples):
    """
    Process a batch from a PyTorch Geometric DataLoader based on clustering criteria.

    Args:
    - batch (Batch): The input batch from a PyTorch Geometric DataLoader.
    - C (int): The number of clusters.
    - num_samples (int): The number of samples to select from one cluster and half of this number from other clusters.

    Returns:
    - Batch: A new batch containing the selected samples.
    """

    # Ensure the batch has the 'cluster_id' attribute
    if not hasattr(batch, 'logits_cluster_id'):
        raise ValueError("Batch must have 'logits_cluster_id' attribute")

    # List to hold selected data
    selected_data_list = []

    # Randomly select one cluster
    selected_cluster = random.randint(0, C-1)

    # Get indices of data in the selected cluster
    indices_in_selected_cluster = (batch.logits_cluster_id == selected_cluster).nonzero(as_tuple=True)[0]

    # Randomly select 'num_samples' data points from the selected cluster
    selected_indices = indices_in_selected_cluster[torch.randperm(len(indices_in_selected_cluster))[:num_samples]]
    selected_data_list.extend([batch[i] for i in selected_indices])

    # Process remaining clusters
    for cluster_id in range(C):
        if cluster_id != selected_cluster:
            # Get indices of data in the current cluster
            indices_in_cluster = (batch.logits_cluster_id == cluster_id).nonzero(as_tuple=True)[0]

            # Randomly select 'num_samples / 2' data points from the current cluster
            num_samples_half = num_samples // 2
            selected_indices = indices_in_cluster[torch.randperm(len(indices_in_cluster))[:num_samples_half]]
            selected_data_list.extend([batch[i] for i in selected_indices])

    # Create a new batch from the selected list of data
    new_batch = Batch.from_data_list(selected_data_list)
    return new_batch


def split_dataset_and_embeddings(val_dataset, val_graph_emb_dict):
    """
    Splits a PyG dataset and a corresponding embeddings dictionary into two parts.

    Args:
    val_dataset (Dataset): The PyG dataset to split.
    val_graph_emb_dict (Dict): The dictionary of graph embeddings.

    Returns:
    Tuple[Dataset, Dataset, Dict, Dict]: Two datasets and two corresponding embedding dictionaries.
    """
    # Ensure reproducibility
    random.seed(0)
    torch.manual_seed(0)

    # Extract indices for each class
    class_indices = {i: [] for i in range(3)}
    for idx, data in enumerate(val_dataset):
        y = data.y.item()
        if y in class_indices:
            class_indices[y].append(idx)

    # Randomly select 500 indices from each class for both splits
    split_indices = [[], []]
    for class_idx in class_indices:
        random.shuffle(class_indices[class_idx])
        split_indices[0].extend(class_indices[class_idx][:500])
        split_indices[1].extend(class_indices[class_idx][500:1000])

    # Create two datasets from the split indices
    datasets = [val_dataset[indices] for indices in split_indices]

    # Split the val_graph_emb_dict according to the indices
    emb_dicts = [{}, {}]
    for i, indices in enumerate(split_indices):
        for key in val_graph_emb_dict:
            emb_dicts[i][key] = val_graph_emb_dict[key][indices]

    return datasets[0], datasets[1], emb_dicts[0], emb_dicts[1]



from torch.utils.data import Dataset


class FineTuneDataset(Dataset):
    def __init__(self, embeddings, labels,svm_proba=None):
        """
        Custom dataset to handle embeddings and labels.

        Args:
        embeddings (torch.Tensor): The embedding matrix of shape (N, D).
        labels (torch.Tensor): The labels of shape (N,).
        """
        self.embeddings = embeddings
        self.labels = labels
        self.svm_proba = svm_proba

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.labels)

    def __getitem__(self, idx):
        """Retrieves the embedding and label at the specified index."""
        if self.svm_proba is not None:
            return idx,self.embeddings[idx], self.labels[idx], self.svm_proba[idx]
        else:
            return self.embeddings[idx], self.labels[idx]
    
    
    
    
    