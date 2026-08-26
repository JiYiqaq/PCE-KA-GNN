from __future__ import annotations

import math

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


class DeviceContextEncoder(nn.Module):
    """Encode scaled numeric conditions and categorical device metadata."""

    def __init__(
        self,
        numeric_context_dim: int,
        category_sizes: tuple[int, ...],
        context_hidden: int,
        grid_feat: int,
        use_bias: bool,
    ) -> None:
        super().__init__()
        if numeric_context_dim < 1 or context_hidden < 1:
            raise ValueError("numeric_context_dim and context_hidden must be positive")
        if any(size < 2 for size in category_sizes):
            raise ValueError("every category vocabulary must include missing and unknown tokens")
        self.numeric_context_dim = int(numeric_context_dim)
        self.category_sizes = tuple(int(size) for size in category_sizes)
        embedding_dims = tuple(min(16, max(2, math.ceil(math.sqrt(size)))) for size in category_sizes)
        self.embeddings = nn.ModuleList(
            nn.Embedding(size, width) for size, width in zip(category_sizes, embedding_dims)
        )
        self.context_layer = KAN_linear(
            numeric_context_dim + sum(embedding_dims),
            context_hidden,
            grid_feat,
            addbias=use_bias,
        )
        self.activation = nn.LeakyReLU()

    def forward(
        self,
        numeric_context: torch.Tensor,
        categorical_context: torch.Tensor,
    ) -> torch.Tensor:
        if numeric_context.ndim != 2 or numeric_context.shape[1] != self.numeric_context_dim:
            raise ValueError("numeric context has an unexpected shape")
        expected_categories = len(self.embeddings)
        if categorical_context.ndim != 2 or categorical_context.shape[1] != expected_categories:
            raise ValueError("categorical context has an unexpected shape")
        if numeric_context.shape[0] != categorical_context.shape[0]:
            raise ValueError("numeric and categorical context batch sizes must match")
        embedded = [
            embedding(categorical_context[:, index])
            for index, embedding in enumerate(self.embeddings)
        ]
        context = torch.cat([numeric_context, *embedded], dim=-1)
        return self.activation(self.context_layer(context))


class DualKAGNNRegressor(nn.Module):
    """Shared KA-GNN encoders fused with optional device/process context."""

    def __init__(
        self,
        in_feat: int = 113,
        hidden_feat: int = 64,
        fusion_hidden: int = 32,
        grid_feat: int = 1,
        num_layers: int = 4,
        pooling: str = "avg",
        use_bias: bool = True,
        numeric_context_dim: int = 0,
        category_sizes: tuple[int, ...] = (),
        context_hidden: int = 32,
        use_context: bool = False,
    ) -> None:
        super().__init__()
        self.use_context = bool(use_context)
        self.encoder = KAGraphEncoder(
            in_feat=in_feat,
            hidden_feat=hidden_feat,
            grid_feat=grid_feat,
            num_layers=num_layers,
            pooling=pooling,
            use_bias=use_bias,
        )
        if self.use_context:
            self.context_encoder = DeviceContextEncoder(
                numeric_context_dim=numeric_context_dim,
                category_sizes=category_sizes,
                context_hidden=context_hidden,
                grid_feat=grid_feat,
                use_bias=use_bias,
            )
        else:
            self.context_encoder = None
        fusion_input_dim = hidden_feat * 4 + (context_hidden if self.use_context else 0)
        self.fusion_layer = KAN_linear(
            fusion_input_dim,
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
        numeric_context: torch.Tensor | None = None,
        categorical_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        donor = self.encoder(donor_graph, donor_graph.ndata["feat"])
        acceptor = self.encoder(acceptor_graph, acceptor_graph.ndata["feat"])
        pair_features = torch.cat(
            [donor, acceptor, torch.abs(donor - acceptor), donor * acceptor],
            dim=-1,
        )
        if self.use_context:
            if numeric_context is None or categorical_context is None:
                raise ValueError("context tensors are required when use_context=True")
            context = self.context_encoder(numeric_context, categorical_context)
            pair_features = torch.cat([pair_features, context], dim=-1)
        hidden = self.activation(self.fusion_layer(pair_features))
        return self.output_layer(hidden).squeeze(-1)
