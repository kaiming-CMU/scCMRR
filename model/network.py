import torch
from torch import nn
from torch.nn import functional as F

from .graph import GraphConv
from .fusion import AttentionFusion
from .relations import relations, relation_loss
from .transport import SemiRelaxedOT


class ViewAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_cells, n_clusters):
        super().__init__()
        self.encoder = GraphConv(input_dim, hidden_dim)
        self.decoder = GraphConv(hidden_dim, input_dim)
        self.affinity = nn.Parameter(torch.randn(n_cells, n_cells) * 0.01)
        self.prototypes = nn.Linear(hidden_dim, n_clusters, bias=False)

    def forward(self, x, adjacency):
        features = self.encoder(x, adjacency)
        latent_reconstruction = self.affinity @ features
        reconstruction = self.decoder(latent_reconstruction, adjacency)
        with torch.no_grad():
            self.prototypes.weight.copy_(F.normalize(self.prototypes.weight, dim=1))
        return features, reconstruction, latent_reconstruction, self.prototypes(features)


class scCMRR(nn.Module):
    def __init__(self, dimensions, hidden_dim, n_cells, n_clusters, rank_k, threshold, ot):
        super().__init__()
        self.views = nn.ModuleList([
            ViewAutoencoder(dim, hidden_dim, n_cells, n_clusters) for dim in dimensions
        ])
        self.fusion = ViewAutoencoder(sum(dimensions), hidden_dim, n_cells, n_clusters)
        self.attention = AttentionFusion(hidden_dim)
        self.rank_k = rank_k
        self.threshold = threshold
        self.transport = SemiRelaxedOT(n_clusters, **ot)

    def set_epoch(self, epoch):
        self.transport.set_epoch(epoch)

    def embeddings(self, x, adjacency, fusion_x, fusion_adjacency):
        fusion = self.fusion.encoder(fusion_x, fusion_adjacency)
        views = [model.encoder(v, a) for model, v, a in zip(self.views, x, adjacency)]
        return fusion, self.attention(views)

    def forward(self, x, adjacency, fusion_x, fusion_adjacency):
        fusion = self.fusion(fusion_x, fusion_adjacency)
        views = [model(v, a) for model, v, a in zip(self.views, x, adjacency)]
        reconstruction = F.mse_loss(fusion[1], fusion_x) + F.mse_loss(fusion[0], fusion[2])
        for view, original in zip(views, x):
            reconstruction = reconstruction + F.mse_loss(view[1], original) + F.mse_loss(view[0], view[2])
        fusion_relation = relations(fusion[3], self.rank_k, self.threshold)[2]
        view_relations = [relations(view[3], self.rank_k, self.threshold) for view in views]
        refinement = relation_loss(fusion_relation, view_relations)
        transport = self.transport(fusion[3], [view[3] for view in views], fusion[0])
        return reconstruction, refinement, transport
