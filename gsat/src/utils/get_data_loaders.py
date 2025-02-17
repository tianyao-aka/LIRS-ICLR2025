import torch
import numpy as np
from torch_geometric.data import Batch
from torch_geometric.utils import degree
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset

from ogb.graphproppred import PygGraphPropPredDataset
# from datasets import SynGraphDataset, Mutag, SPMotif, graph_sst2

import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))

# Add the parent directory (xxx) to the system path
parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
sys.path.append(parent_dir)

import warnings
import random
# from datasets.drugood_dataset import DrugOOD
# from drugood.datasets import build_dataset
from datasets.mnistsp_dataset import CMNIST75sp
from GOOD.data.good_datasets.good_cmnist import GOODCMNIST
from GOOD.data.good_datasets.good_motif import GOODMotif
from GOOD.data.good_datasets.good_hiv import GOODHIV
from mmcv import Config



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

def get_data_loaders(data_dir, dataset_name, batch_size, splits, random_state, mutag_x=False,config_dir = None,domain='scaffold',shift='covariate'):
    num_workers = 2 if torch.cuda.is_available() else 0
    multi_label = False
    # assert dataset_name in ['ba_2motifs', 'mutag', 'Graph-SST2', 'mnist',
    #                         'spmotif_0.5', 'spmotif_0.7', 'spmotif_0.9',
    #                         'ogbg_molhiv', 'ogbg_moltox21', 'ogbg_molbace',
    #                         'ogbg_molbbbp', 'ogbg_molclintox', 'ogbg_molsider']

    if dataset_name == 'ba_2motifs':
        dataset = SynGraphDataset(data_dir, 'ba_2motifs')
        split_idx = get_random_split_idx(dataset, splits)
        loaders, test_set = get_loaders_and_test_set(batch_size, dataset=dataset, split_idx=split_idx)
        train_set = dataset[split_idx["train"]]

    elif dataset_name == 'mutag':
        dataset = Mutag(root=data_dir / 'mutag')
        split_idx = get_random_split_idx(dataset, splits, mutag_x=mutag_x)
        loaders, test_set = get_loaders_and_test_set(batch_size, dataset=dataset, split_idx=split_idx)
        train_set = dataset[split_idx['train']]

    elif 'ogbg' in dataset_name:
        dataset = PygGraphPropPredDataset(root=data_dir, name='-'.join(dataset_name.split('_')))
        if domain == "scaffold":
            split_idx = dataset.get_idx_split()
        else:
            split_idx = size_split_idx(dataset)
        # print('[INFO] Using default splits!')
        # loaders, test_set = get_loaders_and_test_set(batch_size, dataset=dataset, split_idx=split_idx)
        train_set = dataset[split_idx["train"]]
        val_set = dataset[split_idx["valid"]]
        test_set = dataset[split_idx["test"]]

        # num_samples = len(train_set)

        # # Generate shuffled indices
        # indices = torch.randperm(num_samples)

        # # Calculate the split index
        # split_idx = int(num_samples * 0.7)

        # # Split indices into training and test indices
        # train_indices = indices[:split_idx]
        # test_indices = indices[split_idx:]
        # train_dataset = train_set[train_indices]
        # test_dataset = train_set[test_indices]
        print('[INFO] Using train splits!')
        loaders, test_set = get_loaders_and_test_set(batch_size, dataset_splits={'train': train_set, 'valid': val_set, 'test': test_set,'entire_train':train_set})
    
    elif 'goodhiv' in dataset_name.lower():
        dataset, meta_info = GOODHIV.load("GOOD_data/", domain=domain, shift=shift, generate=False)
        train_set = dataset['train']
        val_set = dataset['val']
        test_set = dataset['test']
        print('[INFO] GOODHIV')
        loaders, test_set = get_loaders_and_test_set(batch_size, dataset_splits={'train': train_set, 'valid': val_set, 'test': test_set,'entire_train':train_set})

    elif 'good' in dataset_name.lower() and 'motif' in dataset_name.lower():
        dataset, meta_info = GOODMotif.load("GOOD_data/", domain=domain, shift=shift, generate=False)
        train_set = dataset['train']
        val_set = dataset['val']
        test_set = dataset['test']
        print('[INFO] GOODMOTIF')
        loaders, test_set = get_loaders_and_test_set(batch_size, dataset_splits={'train': train_set, 'valid': val_set, 'test': test_set,'entire_train':train_set})

    elif 'cmnist' in dataset_name.lower():
        dataset, meta_info = GOODCMNIST.load("GOOD_data/", domain='color', shift='covariate', generate=False)
        train_set = dataset['train']
        val_set = dataset['val']
        test_set = dataset['test']
        print('[INFO] GOOD CMNIST')
        loaders, test_set = get_loaders_and_test_set(batch_size, dataset_splits={'train': train_set, 'valid': val_set, 'test': test_set,'entire_train':train_set})
        
        
    elif 'sst2' in dataset_name.lower():
        dataset = graph_sst2.get_dataset(dataset_dir=data_dir, dataset_name='Graph-SST2', task=None)
        loaders, (train_set, valid_set, test_set) = graph_sst2.get_dataloader(dataset, batch_size=batch_size, degree_bias=True, seed=random_state)
        print('[INFO] Using default splits!')
        print ('dataset info SST2: ',len(train_set),len(valid_set),len(test_set))
        loaders['entire_train'] = DataLoader(valid_set, batch_size=1, shuffle=False)
        loaders['valid'] = loaders['eval']
        # avg = np.mean(np.asarray([train_set[i].x.shape[0] for i in range(len(train_set))]))
        # print (avg, 'degrees')
    

    elif 'spmotif' in dataset_name:
        b = float(dataset_name.split('_')[-1])
        train_set = SPMotif(root=data_dir / dataset_name, b=b, mode='train')
        valid_set = SPMotif(root=data_dir / dataset_name, b=b, mode='val')
        test_set = SPMotif(root=data_dir / dataset_name, b=b, mode='test')
        indices = torch.randperm(len(train_set))
        # Calculate the number of training samples
        train_size = int(len(train_set) * 0.7)
        # Split indices into training and test sets
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]
        # Create training and test datasets using the selected indices
        train_set_new = train_set[train_indices]
        test_set_new = train_set[test_indices]
        print('[INFO] Using train splits!')
        print (f'{len(train_set_new)},{len(test_set_new)}')
        loaders, test_set = get_loaders_and_test_set(batch_size, dataset_splits={'train': train_set_new, 'valid': test_set_new, 'test': test_set_new})


    elif 'drugood' in dataset_name:
        config_path = os.path.join("../src/","drugood_configs", dataset_name + ".py")
        cfg = Config.fromfile(config_path)
        root = os.path.join(data_dir,"DrugOOD")
        print (root)
        print (cfg.data.train)
        print (root)
        train_set = DrugOOD(root=root, dataset=build_dataset(cfg.data.train), name=dataset_name, mode="train")
        indices = torch.randperm(len(train_set))
        # Calculate the number of training samples
        train_size = int(len(train_set) * 0.7)
        # Split indices into training and test sets
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]
        # Create training and test datasets using the selected pindices
        train_set_new = train_set[train_indices]
        test_set_new = train_set[test_indices]
        print('[INFO] Using train splits for DrugOOD!')
        loaders, test_set = get_loaders_and_test_set(batch_size, dataset_splits={'train': train_set_new, 'valid': test_set_new, 'test': test_set_new,'entire_train':train_set})


    # elif dataset_name == 'cmnist':
    #     n_val_data = 5000
    #     train_set = CMNIST75sp(os.path.join(data_dir, 'CMNISTSP/'), mode='train')
    #     test_set = CMNIST75sp(os.path.join(data_dir, 'CMNISTSP/'), mode='test')
    #     perm_idx = torch.randperm(len(test_set), generator=torch.Generator().manual_seed(0))
    #     test_val = test_set[perm_idx]
    #     val_set, test_set = test_val[:n_val_data], test_val[n_val_data:]
    #     print('[INFO] Using train splits for CMNIST75sp!')
    #     loaders, test_set = get_loaders_and_test_set(batch_size, dataset_splits={'train': train_set, 'valid': val_set, 'test': val_set,'entire_train':train_set})


    x_dim = test_set[0].x.shape[1]
    try:
        edge_attr_dim = 0 if test_set[0].edge_attr is None else test_set[0].edge_attr.shape[1]
    except:
        edge_attr_dim = 0
    if isinstance(test_set, list):
        num_class = Batch.from_data_list(test_set).y.unique().shape[0]
    elif test_set.data.y.shape[-1] == 1 or len(test_set.data.y.shape) == 1:
        num_class = test_set.data.y.unique().shape[0]
    else:
        num_class = test_set.data.y.shape[-1]
        multi_label = True

    print('[INFO] Calculating degree...')
    # Compute in-degree histogram over training data.
    # deg = torch.zeros(10, dtype=torch.long)
    # for data in train_set:
    #     d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
    #     deg += torch.bincount(d, minlength=deg.numel())
    batched_train_set = Batch.from_data_list(train_set)
    d = degree(batched_train_set.edge_index[1], num_nodes=batched_train_set.num_nodes, dtype=torch.long)
    deg = torch.bincount(d, minlength=10)

    aux_info = {'deg': deg, 'multi_label': multi_label}
    return loaders, test_set, x_dim, edge_attr_dim, num_class, aux_info


def get_random_split_idx(dataset, splits, random_state=None, mutag_x=False):
    if random_state is not None:
        np.random.seed(random_state)

    print('[INFO] Randomly split dataset!')
    idx = np.arange(len(dataset))
    np.random.shuffle(idx)

    if not mutag_x:
        n_train, n_valid = int(splits['train'] * len(idx)), int(splits['valid'] * len(idx))
        train_idx = idx[:n_train]
        valid_idx = idx[n_train:n_train+n_valid]
        test_idx = idx[n_train+n_valid:]
    else:
        print('[INFO] mutag_x is True!')
        n_train = int(splits['train'] * len(idx))
        train_idx, valid_idx = idx[:n_train], idx[n_train:]
        test_idx = [i for i in range(len(dataset)) if (dataset[i].y.squeeze() == 0 and dataset[i].edge_label.sum() > 0)]
    return {'train': train_idx, 'valid': valid_idx, 'test': test_idx}


def get_loaders_and_test_set(batch_size, dataset=None, split_idx=None, dataset_splits=None):
    num_workers = 2 if torch.cuda.is_available() else 0
    if split_idx is not None:
        assert dataset is not None
        train_loader = DataLoader(dataset[split_idx["train"]], batch_size=batch_size, shuffle=True,num_workers=num_workers)
        valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=batch_size, shuffle=False,num_workers=num_workers)
        test_loader = DataLoader(dataset[split_idx["test"]], batch_size=batch_size, shuffle=False,num_workers=num_workers)
        test_set = dataset.copy(split_idx["test"])  # For visualization
    else:
        assert dataset_splits is not None
        train_loader = DataLoader(dataset_splits['train'], batch_size=batch_size, shuffle=True,num_workers=num_workers)
        valid_loader = DataLoader(dataset_splits['valid'], batch_size=batch_size, shuffle=False,num_workers=num_workers)
        test_loader = DataLoader(dataset_splits['test'], batch_size=batch_size, shuffle=False,num_workers=num_workers)
        entire_train_loader = DataLoader(dataset_splits['entire_train'], batch_size=1, shuffle=False,num_workers=num_workers)
        test_set = dataset_splits['test']  # For visualization
    return {'train': train_loader, 'valid': valid_loader, 'test': test_loader,'entire_train':entire_train_loader}, test_set



