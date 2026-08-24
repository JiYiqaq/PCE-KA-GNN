from __future__ import annotations

import dgl
import torch
import torch.nn as nn
from dgl.nn import AvgPooling, MaxPooling, SumPooling

from model.ka_gnn import KAN_linear, NaiveFourierKANLayer


class KAGraphEncoder(nn.Module):
    """Graph encoder using the Fourier KAN layers from the original KA-GNN."""

    def __init__(
        self,
        in_feat: int,
        hidden_feat: int,
        grid_feat: int,
        num_layers: int,
        pooling: str,
        use_bias: bool = True,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        if pooling not in {"avg", "sum", "max"}:
            raise ValueError("pooling must be one of: avg, sum, max")

        self.pooling = pooling
        self.input_layer = KAN_linear(in_feat, hidden_feat, grid_feat, addbias=use_bias)
        self.message_layers = nn.ModuleList(
            NaiveFourierKANLayer(
                hidden_feat,
                hidden_feat,
                grid_feat,
                addbias=use_bias,
            )
            for _ in range(num_layers - 1)
        )
        self.poolers = nn.ModuleDict(
            {
                "avg": AvgPooling(),
                "sum": SumPooling(),
                "max": MaxPooling(),
            }
        )

    def forward(self, graph: dgl.DGLGraph, features: torch.Tensor) -> torch.Tensor:
        hidden = self.input_layer(features)
        for layer in self.message_layers:
            hidden = layer(graph, hidden)
        return self.poolers[self.pooling](graph, hidden)


class DualKAGNNRegressor(nn.Module):
    """Shared KA-GNN encoder plus an ordered donor-acceptor regression head."""

    def __init__(
        self,
        in_feat: int = 113,
        hidden_feat: int = 64,
        fusion_hidden: int = 32,
        grid_feat: int = 1,
        num_layers: int = 4,
        pooling: str = "avg",
        use_bias: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = KAGraphEncoder(
            in_feat=in_feat,
            hidden_feat=hidden_feat,
            grid_feat=grid_feat,
            num_layers=num_layers,
            pooling=pooling,
            use_bias=use_bias,
        )
        self.fusion_layer = KAN_linear(
            hidden_feat * 4,
            fusion_hidden,
            grid_feat,
            addbias=use_bias,
        )
        self.activation = nn.LeakyReLU()
        self.output_layer = KAN_linear(
            fusion_hidden,
            1,
            grid_feat,
            addbias=True,
        )

    def forward(
        self,
        donor_graph: dgl.DGLGraph,
        acceptor_graph: dgl.DGLGraph,
    ) -> torch.Tensor:
        donor = self.encoder(donor_graph, donor_graph.ndata["feat"])
        acceptor = self.encoder(acceptor_graph, acceptor_graph.ndata["feat"])
        pair_features = torch.cat(
            [donor, acceptor, torch.abs(donor - acceptor), donor * acceptor],
            dim=-1,
        )
        hidden = self.activation(self.fusion_layer(pair_features))
        return self.output_layer(hidden).squeeze(-1)
