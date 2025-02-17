# https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/basic_gnn.html#GIN
import torch
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool
# from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from GOOD.utils.data import x_map, e_map
from .conv_layers import GINConv, GINEConv


class AtomEncoder(torch.nn.Module):
    r"""
    atom (node) feature encoding specified for molecule data.

    Args:
        emb_dim: number of dimensions of embedding
    """

    def __init__(self, emb_dim):

        super(AtomEncoder, self).__init__()

        self.atom_embedding_list = torch.nn.ModuleList()

        feat_dims = list(map(len, x_map.values()))

        for i, dim in enumerate(feat_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, x):
        r"""
        atom (node) feature encoding specified for molecule data.

        Args:
            x (Tensor): node features

        Returns (Tensor):
            atom (node) embeddings
        """
        
        x_embedding = 0
        for i in range(x.shape[1]):
            x_embedding += self.atom_embedding_list[i](x[:, i])

        return x_embedding


class BondEncoder(torch.nn.Module):
    r"""
    bond (edge) feature encoding specified for molecule data.

    Args:
        emb_dim: number of dimensions of embedding
    """

    def __init__(self, emb_dim):
        super(BondEncoder, self).__init__()

        self.bond_embedding_list = torch.nn.ModuleList()

        edge_feat_dims = list(map(len, e_map.values()))

        for i, dim in enumerate(edge_feat_dims):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)

    def forward(self, edge_attr):
        r"""
        bond (edge) feature encoding specified for molecule data.

        Args:
            edge_attr (Tensor): edge attributes

        Returns (Tensor):
            bond (edge) embeddings

        """
        bond_embedding = 0
        for i in range(edge_attr.shape[1]):
            bond_embedding += self.bond_embedding_list[i](edge_attr[:, i])

        return bond_embedding

class GIN(nn.Module):
    def __init__(self, x_dim, edge_attr_dim, num_class, multi_label, model_config):
        super().__init__()

        self.n_layers = model_config['n_layers']
        hidden_size = model_config['hidden_size']
        self.edge_attr_dim = edge_attr_dim
        self.dropout_p = model_config['dropout_p']
        self.use_edge_attr = model_config.get('use_edge_attr', True)
        self.ogbg_dataset = False

        if model_config.get('atom_encoder', False):
            self.ogbg_dataset = True
            self.node_encoder = AtomEncoder(emb_dim=hidden_size)
            if edge_attr_dim != 0 and self.use_edge_attr:
                self.edge_encoder = BondEncoder(emb_dim=hidden_size)
        else:
            self.node_encoder = Linear(x_dim, hidden_size)
            if edge_attr_dim != 0 and self.use_edge_attr:
                self.edge_encoder = Linear(edge_attr_dim, hidden_size)

        self.convs = nn.ModuleList()
        self.relu = nn.ReLU()
        self.pool = global_add_pool

        for _ in range(self.n_layers):
            if edge_attr_dim != 0 and self.use_edge_attr:
                self.convs.append(GINEConv(GIN.MLP(hidden_size, hidden_size), edge_dim=hidden_size))
            else:
                self.convs.append(GINConv(GIN.MLP(hidden_size, hidden_size)))

        self.fc_out = nn.Sequential(nn.Linear(hidden_size, 1 if num_class == 2 and not multi_label else num_class))

    def forward(self, x, edge_index, batch, edge_attr=None, edge_atten=None):
        if self.ogbg_dataset:
            x = x.long()
        x = self.node_encoder(x)
        if edge_attr is not None and self.use_edge_attr and self.edge_attr_dim>0:
            if self.ogbg_dataset:
                edge_attr = edge_attr.long()
            edge_attr = self.edge_encoder(edge_attr)

        for i in range(self.n_layers):
            x = self.convs[i](x, edge_index, edge_attr=edge_attr, edge_atten=edge_atten)
            x = self.relu(x)
            x = F.dropout(x, p=self.dropout_p, training=self.training)
        return self.fc_out(self.pool(x, batch))

    @staticmethod
    def MLP(in_channels: int, out_channels: int):
        return nn.Sequential(
            Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            Linear(out_channels, out_channels),
        )

    def get_emb(self, x, edge_index, batch, edge_attr=None, edge_atten=None):
        if self.ogbg_dataset:
            x = x.long()
        x = self.node_encoder(x)
        if edge_attr is not None and self.use_edge_attr and self.edge_attr_dim>0:
            if self.ogbg_dataset:
                edge_attr = edge_attr.long()
            edge_attr = self.edge_encoder(edge_attr)

        for i in range(self.n_layers):
            x = self.convs[i](x, edge_index, edge_attr=edge_attr, edge_atten=edge_atten)
            x = self.relu(x)
            x = F.dropout(x, p=self.dropout_p, training=self.training)
        return x

    def get_pred_from_emb(self, emb, batch):
        return self.fc_out(self.pool(emb, batch))
