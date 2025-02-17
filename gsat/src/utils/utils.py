import torch
import random
import numpy as np
import networkx as nx
from rdkit import Chem
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, sort_edge_index
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams


init_metric_dict = {'metric/best_clf_epoch': 0, 'metric/best_clf_valid_loss': 0,
                    'metric/best_clf_train': 0, 'metric/best_clf_valid': 0, 'metric/best_clf_test': 0,
                    'metric/best_x_roc_train': 0, 'metric/best_x_roc_valid': 0, 'metric/best_x_roc_test': 0,
                    'metric/best_x_precision_train': 0, 'metric/best_x_precision_valid': 0, 'metric/best_x_precision_test': 0}


def reorder_like(from_edge_index, to_edge_index, values):
    from_edge_index, values = sort_edge_index(from_edge_index, values)
    ranking_score = to_edge_index[0] * (to_edge_index.max()+1) + to_edge_index[1]
    ranking = ranking_score.argsort().argsort()
    if not (from_edge_index[:, ranking] == to_edge_index).all():
        raise ValueError("Edges in from_edge_index and to_edge_index are different, impossible to match both.")
    return values[ranking]


def process_data(data, use_edge_attr):
    if not use_edge_attr:
        data.edge_attr = None
    if data.get('edge_label', None) is None:
        data.edge_label = torch.zeros(data.edge_index.shape[1])
    return data


def load_checkpoint(model, model_dir, model_name, map_location=None):
    checkpoint = torch.load(model_dir / (model_name + '.pt'), map_location=map_location)
    model.load_state_dict(checkpoint['model_state_dict'])


def save_checkpoint(model, model_dir, model_name):
    torch.save({'model_state_dict': model.state_dict()}, model_dir / (model_name + '.pt'))


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def get_local_config_name(model_name, dataset_name):
    if 'ogbg_mol' in dataset_name:
        local_config_name = f'{model_name}-ogbg_mol.yml'
    elif 'spmotif' in dataset_name:
        local_config_name = f'{model_name}-spmotif.yml'
    else:
        local_config_name = f'{model_name}-{dataset_name}.yml'
    return local_config_name


def write_stat_from_metric_dicts(hparam_dict, metric_dicts, writer):
    res = {metric: {'value': [], 'mean': 0, 'std': 0} for metric in metric_dicts[0].keys()}

    for metric_dict in metric_dicts:
        for metric, value in metric_dict.items():
            res[metric]['value'].append(value)

    stat = {}
    for metric, value in res.items():
        stat[metric] = np.mean(value['value'])
        stat[metric+'/std'] = np.std(value['value'])

    writer.add_hparams(hparam_dict, stat)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class Writer(SummaryWriter):
    def add_hparams(
        self, hparam_dict, metric_dict, hparam_domain_discrete=None, run_name=None
    ):
        torch._C._log_api_usage_once("tensorboard.logging.add_hparams")
        if type(hparam_dict) is not dict or type(metric_dict) is not dict:
            raise TypeError('hparam_dict and metric_dict should be dictionary.')
        exp, ssi, sei = hparams(hparam_dict, metric_dict, hparam_domain_discrete)

        logdir = self._get_file_writer().get_logdir()
        with SummaryWriter(log_dir=logdir) as w_hp:
            w_hp.file_writer.add_summary(exp)
            w_hp.file_writer.add_summary(ssi)
            w_hp.file_writer.add_summary(sei)
            for k, v in metric_dict.items():
                w_hp.add_scalar(k, v)


def visualize_a_graph(edge_index, edge_att, node_label, dataset_name, coor=None, norm=False, mol_type=None, nodesize=300):
    plt.clf()
    if norm:
        edge_att = edge_att**10
        edge_att = (edge_att - edge_att.min()) / (edge_att.max() - edge_att.min() + 1e-6)

    if mol_type is None or dataset_name == 'Graph-SST2':
        atom_colors = {0: '#E49D1C', 1: '#FF5357', 2: '#a1c569', 3: '#69c5ba'}
        node_colors = [None for _ in range(node_label.shape[0])]
        for y_idx in range(node_label.shape[0]):
            node_colors[y_idx] = atom_colors[node_label[y_idx].int().tolist()]
    else:
        node_color = ['#29A329', 'lime', '#F0EA00',  'maroon', 'brown', '#E49D1C', '#4970C6', '#FF5357']
        element_idxs = {k: Chem.PeriodicTable.GetAtomicNumber(Chem.GetPeriodicTable(), v) for k, v in mol_type.items()}
        node_colors = [node_color[(v - 1) % len(node_color)] for k, v in element_idxs.items()]

    data = Data(edge_index=edge_index, att=edge_att, y=node_label, num_nodes=node_label.size(0)).to('cpu')
    G = to_networkx(data, node_attrs=['y'], edge_attrs=['att'])

    # calculate Graph positions
    if coor is None:
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = {idx: each.tolist() for idx, each in enumerate(coor)}

    ax = plt.gca()
    for source, target, data in G.edges(data=True):
        ax.annotate(
            '', xy=pos[target], xycoords='data', xytext=pos[source],
            textcoords='data', arrowprops=dict(
                arrowstyle="->" if dataset_name == 'Graph-SST2' else '-',
                lw=max(data['att'], 0) * 3,
                alpha=max(data['att'], 0),  # alpha control transparency
                color='black',  # color control color
                shrinkA=np.sqrt(nodesize) / 2.0 + 1,
                shrinkB=np.sqrt(nodesize) / 2.0 + 1,
                connectionstyle='arc3,rad=0.4' if dataset_name == 'Graph-SST2' else 'arc3'
            ))

    if mol_type is not None:
        nx.draw_networkx_labels(G, pos, mol_type, ax=ax)

    if dataset_name != 'Graph-SST2':
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=nodesize, ax=ax)
        nx.draw_networkx_edges(G, pos, width=1, edge_color='gray', arrows=False, alpha=0.1, ax=ax)
    else:
        nx.draw_networkx_edges(G, pos, width=1, edge_color='gray', arrows=False, alpha=0.1, ax=ax, connectionstyle='arc3,rad=0.4')

    fig = plt.gcf()
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return fig, image



def fidelity(target: torch.Tensor, logits: torch.Tensor, logits_removed_subgraph: torch.Tensor,binary=True):
    """
    Calculates the fidelity scores based on the given target labels, prediction logits, 
    and prediction logits after the removal of the explanatory subgraph.

    Args:
        target (torch.Tensor): A tensor of shape (N,) containing the target labels.
        logits (torch.Tensor): A tensor of shape (N, C) containing the prediction logits.
        logits_removed_subgraph (torch.Tensor): A tensor of shape (N, C) containing the 
                                                 prediction logits after removal of the subgraph.

    Returns:
        Tuple[float, float]: The fidelity+ scores.
    """
    # Get the predicted labels from logits
    if not binary:
        predicted_labels = torch.argmax(logits, dim=1)
        predicted_labels_removed = torch.argmax(logits_removed_subgraph, dim=1)

        # Calculate the fidelity+ score
        correct_predictions = (predicted_labels == target).float()
        correct_predictions_removed = (predicted_labels_removed == target).float()
        
        fidelity_plus = torch.mean(torch.abs(correct_predictions - correct_predictions_removed)).item()

        return fidelity_plus
    
    else:
        predicted_labels = logits>0.5
        predicted_labels = predicted_labels*1.0
        predicted_labels_removed = logits_removed_subgraph>0.5
        predicted_labels_removed = predicted_labels_removed*1.0

        # Calculate the fidelity+ score
        correct_predictions = (predicted_labels == target).float()
        correct_predictions_removed = (predicted_labels_removed == target).float()
        
        fidelity_plus = torch.mean(torch.abs(correct_predictions - correct_predictions_removed)).item()

        return fidelity_plus

def remove_topk_edges(attn_score, batch, topK):
    """
    Given an attention score tensor, edge indices, and a value topK, this function removes the
    top K edges with the highest attention scores and returns the new edge_index.

    Args:
        attn_score (torch.Tensor): A tensor of shape (E,) containing the attention scores for each edge.
        edge_index (torch.Tensor): A tensor of shape (2, E) containing the edge indices.
        topK (int): The number of top attention scores to consider for removal.

    Returns:
        torch.Tensor: A tensor containing the new edge indices after the removal.
    """
    # Get the indices of the top K largest values in attn_score
    edge_index = batch.edge_index
    if 'edge_attr' in batch:
        edge_attr = batch.edge_attr
    else:
        edge_attr = None
    topk_indices = torch.topk(attn_score, topK).indices

    # Create a mask to keep all edges except the top K
    mask = torch.ones(attn_score.size(0), dtype=torch.bool)
    mask[topk_indices] = False

    # Apply the mask to edge_index to get the new edge_index
    new_edge_index = edge_index[:, mask]
    if edge_attr is not None:
        new_edge_attr = edge_attr[mask]
    else:
        new_edge_attr = None

    return new_edge_index,new_edge_attr


def topk_nodes_from_attn(attn_score: torch.Tensor, edge_index: torch.Tensor, topK: int) -> torch.Tensor:
    """
    Given an attention score tensor, edge indices, and a value topK, this function returns the 
    top K nodes with the highest attention scores.

    Args:
        attn_score (torch.Tensor): A tensor of shape (E,) containing the attention scores for each edge.
        edge_index (torch.Tensor): A tensor of shape (2, E) containing the edge indices.
        topK (int): The number of top attention scores to consider.

    Returns:
        torch.Tensor: A tensor containing the nodes corresponding to the top K attention scores.
    """
    # Get the indices of the top K largest values in attn_score
    topk_indices = torch.topk(attn_score, topK).indices

    # Get the corresponding edges from edge_index using the top K indices
    topk_edges = edge_index[:, topk_indices]
    
    # Get the unique nodes from the top K edges
    topk_nodes = torch.unique(topk_edges)

    return topk_nodes


def topk_edges_from_attn(attn_score: torch.Tensor, 
                         edge_index: torch.Tensor, 
                         topK: int) -> torch.Tensor:
    """
    Selects the top-K edges (undirected) based on the attention scores, removing duplicates.

    Args:
        attn_score (torch.Tensor): A tensor of shape (E,) containing the attention scores for each edge.
        edge_index (torch.Tensor): A tensor of shape (2, E) containing directed edge indices.
        topK (int): The number of top attention scores to select.

    Returns:
        torch.Tensor: A tensor of shape (2, K') containing the top-K' edges (K' <= K if fewer 
                      than K unique edges exist) in undirected form, i.e. no duplicates of 
                      (u -> v) and (v -> u).
    """
    # 1. Sort edges by descending attention score
    sorted_indices = torch.argsort(attn_score, descending=True)

    # 2. Keep track of unique undirected edges by storing (min(u, v), max(u, v)) in a set
    selected_indices = []
    seen_pairs = set()  # to avoid duplicates

    for idx in sorted_indices:
        u = edge_index[0, idx].item()
        v = edge_index[1, idx].item()
        pair = (min(u, v), max(u, v))  # undirected form
        if pair not in seen_pairs:
            seen_pairs.add(pair)
            selected_indices.append(idx)
            if len(selected_indices) == topK:
                break

    # 3. Gather the edges corresponding to these indices
    if len(selected_indices) == 0:
        return torch.empty((2, 0), dtype=edge_index.dtype, device=edge_index.device)
    topk_indices_tensor = torch.tensor(selected_indices, dtype=torch.long, device=edge_index.device)
    topk_edges = edge_index[:, topk_indices_tensor]
    return topk_edges