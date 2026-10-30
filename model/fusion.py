import torch
from torch import nn


class AttentionFusion(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.Tanh(),
            nn.Linear(16, 1, bias=False),
        )

    def forward(self, features):
        features = torch.stack(features, dim=1)
        weights = torch.softmax(self.project(features), dim=1)
        return (weights * features).sum(1)
