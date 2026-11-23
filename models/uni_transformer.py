"""Equivariant features and coordinate updates for the rigid SeFMol backbone."""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import knn_graph

from models.common import GaussianSmearing, MLP, outer_product
from models.segment_ops import segment_softmax, segment_sum


def aggregate(logits, messages, target, node_count):
    weights = segment_softmax(logits, target, dim_size=node_count)
    return segment_sum(weights.unsqueeze(-1) * messages, target, dim_size=node_count)


class BaseX2HAttLayer(nn.Module):
    """Update node features using relative coordinates and neighboring features."""

    def __init__(self, hidden_dim, n_heads, edge_feat_dim, distance_dim, act_fn, norm):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        input_dim = 2 * hidden_dim + edge_feat_dim + distance_dim
        self.hk_func = MLP(input_dim, hidden_dim, hidden_dim, norm=norm, act_fn=act_fn)
        self.hv_func = MLP(input_dim, hidden_dim, hidden_dim, norm=norm, act_fn=act_fn)
        self.hq_func = MLP(hidden_dim, hidden_dim, hidden_dim, norm=norm, act_fn=act_fn)

    def forward(self, h, distance, edge_type, source, target, edge_weight):
        inputs = torch.cat((distance, h[target], h[source]), dim=-1)
        inputs = torch.cat((edge_type, inputs), dim=-1)
        keys = self.hk_func(inputs).view(-1, self.n_heads, self.head_dim)
        values = self.hv_func(inputs) * edge_weight.view(-1, 1)
        values = values.view(-1, self.n_heads, self.head_dim)
        queries = self.hq_func(h).view(-1, self.n_heads, self.head_dim)
        logits = (queries[target] * keys / np.sqrt(keys.shape[-1])).sum(-1)
        output = aggregate(logits, values, target, len(h))
        return output.view(-1, self.hidden_dim) + h


class BaseH2XAttLayer(nn.Module):
    """Update coordinates by weighting relative displacement vectors."""

    def __init__(self, hidden_dim, n_heads, edge_feat_dim, distance_dim, act_fn, norm):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        input_dim = 2 * hidden_dim + edge_feat_dim + distance_dim
        self.xk_func = MLP(input_dim, hidden_dim, hidden_dim, norm=norm, act_fn=act_fn)
        self.xv_func = MLP(input_dim, n_heads, hidden_dim, norm=norm, act_fn=act_fn)
        self.xq_func = MLP(hidden_dim, hidden_dim, hidden_dim, norm=norm, act_fn=act_fn)

    def forward(self, h, displacement, distance, edge_type, source, target, edge_weight):
        inputs = torch.cat((distance, h[target], h[source]), dim=-1)
        inputs = torch.cat((edge_type, inputs), dim=-1)
        keys = self.xk_func(inputs).view(-1, self.n_heads, self.head_dim)
        values = self.xv_func(inputs) * edge_weight.view(-1, 1)
        values = values.unsqueeze(-1) * displacement.unsqueeze(1)
        queries = self.xq_func(h).view(-1, self.n_heads, self.head_dim)
        logits = (queries[target] * keys / np.sqrt(keys.shape[-1])).sum(-1)
        return aggregate(logits, values, target, len(h)).mean(1)


class AttentionLayerO2TwoUpdateNodeGeneral(nn.Module):
    def __init__(self, hidden_dim, n_heads, num_r_gaussian, edge_feat_dim, act_fn, norm,
                 num_x2h, num_h2x, r_max):
        super().__init__()
        self.distance_expansion = GaussianSmearing(0.0, r_max, num_gaussians=num_r_gaussian)
        layer_args = (hidden_dim, n_heads, edge_feat_dim, num_r_gaussian * 4, act_fn, norm)
        self.x2h_layers = nn.ModuleList(BaseX2HAttLayer(*layer_args) for _ in range(num_x2h))
        self.h2x_layers = nn.ModuleList(BaseH2XAttLayer(*layer_args) for _ in range(num_h2x))

    def forward(self, h, x, edge_type, edge_index, ligand_mask, edge_weight):
        source, target = edge_index
        displacement = x[target] - x[source]
        distance = torch.norm(displacement, p=2, dim=-1, keepdim=True)
        distance_features = outer_product(edge_type, self.distance_expansion(distance))
        for layer in self.x2h_layers:
            h = layer(h, distance_features, edge_type, source, target, edge_weight)
        for layer in self.h2x_layers:
            distance_features = outer_product(edge_type, self.distance_expansion(distance))
            delta = layer(h, displacement, distance_features, edge_type, source, target, edge_weight)
            x = x + delta * ligand_mask[:, None]
            displacement = x[target] - x[source]
            distance = torch.norm(displacement, p=2, dim=-1, keepdim=True)
        return h, x


class UniTransformerO2TwoUpdateGeneral(nn.Module):
    def __init__(self, num_blocks, num_layers, hidden_dim, n_heads, k, num_r_gaussian,
                 edge_feat_dim, act_fn, norm, num_x2h, num_h2x, r_max,
                 safe_knn=False):
        super().__init__()
        self.num_blocks = num_blocks
        self.k = k
        self.safe_knn = safe_knn
        self.distance_expansion = GaussianSmearing(0.0, r_max, num_gaussians=num_r_gaussian)
        self.edge_pred_layer = MLP(num_r_gaussian, 1, hidden_dim)
        # The released state dictionaries include this registered SeFMol layer.
        layer_args = hidden_dim, n_heads, num_r_gaussian, edge_feat_dim, act_fn, norm
        self.init_h_emb_layer = AttentionLayerO2TwoUpdateNodeGeneral(
            *layer_args, num_x2h=1, num_h2x=0, r_max=r_max)
        self.base_block = nn.ModuleList(
            AttentionLayerO2TwoUpdateNodeGeneral(
                *layer_args, num_x2h=num_x2h, num_h2x=num_h2x, r_max=r_max)
            for _ in range(num_layers)
        )

    def _safe_knn_graph(self, x, batch):
        """Build per-complex KNN with standard PyTorch operations.

        Some torch_cluster CUDA builds intermittently fault on large,
        variable-size CrossDocked batches even with valid finite coordinates.
        This path preserves the exact KNN graph definition while avoiding that
        custom CUDA kernel.  Neighbour selection is intentionally detached,
        matching the non-differentiable graph construction in torch_cluster.
        """
        if batch is None:
            batch = x.new_zeros((x.size(0),), dtype=torch.long)
        batch = batch.to(device=x.device, dtype=torch.long).view(-1)
        if batch.numel() != x.size(0):
            raise ValueError('KNN batch must contain one entry per node')
        edge_parts = []
        with torch.no_grad():
            for graph_id in torch.unique(batch, sorted=True):
                node_index = torch.nonzero(
                    batch == graph_id, as_tuple=False).view(-1)
                node_count = int(node_index.numel())
                neighbour_count = min(int(self.k), max(node_count - 1, 0))
                if neighbour_count == 0:
                    continue
                distance = torch.cdist(x[node_index], x[node_index])
                distance.fill_diagonal_(float('inf'))
                neighbour = torch.topk(
                    distance, k=neighbour_count, largest=False, dim=-1).indices
                source = node_index[neighbour.reshape(-1)]
                target = node_index.repeat_interleave(neighbour_count)
                edge_parts.append(torch.stack([source, target], dim=0))
        if not edge_parts:
            return torch.empty((2, 0), device=x.device, dtype=torch.long)
        return torch.cat(edge_parts, dim=1)

    def _connect_edge(self, x, batch):
        if self.safe_knn:
            return self._safe_knn_graph(x, batch)
        node_order = torch.argsort(batch.view(-1))
        knn_x, knn_batch = x[node_order], batch[node_order]
        force_cpu = os.environ.get('SEFMOL_FORCE_CPU_KNN', '').lower() in {'1', 'true', 'yes', 'on'}
        if force_cpu and x.is_cuda:
            edge_index = knn_graph(knn_x.detach().cpu(), k=self.k, batch=knn_batch.detach().cpu(),
                                   flow='source_to_target').to(x.device)
        else:
            edge_index = knn_graph(knn_x, k=self.k, batch=knn_batch, flow='source_to_target')
        return node_order[edge_index]

    @staticmethod
    def _build_edge_type(edge_index, mask_ligand):
        src, dst = edge_index
        edge_type = torch.zeros(len(src)).to(edge_index)
        n_src = mask_ligand[src] == 1
        n_dst = mask_ligand[dst] == 1
        edge_type[n_src & n_dst] = 0
        edge_type[n_src & ~n_dst] = 1
        edge_type[~n_src & n_dst] = 2
        edge_type[~n_src & ~n_dst] = 3
        edge_type = F.one_hot(edge_type, num_classes=4)
        return edge_type

    def forward(self, h, x, ligand_mask, batch):
        for _ in range(self.num_blocks):
            edge_index = self._connect_edge(x, batch)
            if edge_index.size(1) == 0:
                continue
            source, target = edge_index
            edge_type = self._build_edge_type(edge_index, ligand_mask)
            distance = torch.norm(x[target] - x[source], p=2, dim=-1, keepdim=True)
            edge_weight = torch.sigmoid(self.edge_pred_layer(self.distance_expansion(distance)))
            for layer in self.base_block:
                h, x = layer(h, x, edge_type, edge_index, ligand_mask, edge_weight)
        return {'x': x, 'h': h}
