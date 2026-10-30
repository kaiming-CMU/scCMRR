import torch
from torch import nn


class GraphConv(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x, adjacency):
        graph = adjacency.clone()
        graph.fill_diagonal_(1)
        inverse_degree = graph.sum(1).clamp_min(1e-12).rsqrt()
        graph = inverse_degree[:, None] * graph * inverse_degree[None, :]
        return graph @ self.linear(x) + self.bias
