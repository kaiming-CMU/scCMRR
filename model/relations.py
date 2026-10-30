import torch
from torch.nn import functional as F


def relations(logits, rank_k, threshold):
    normalized = F.normalize(logits, dim=-1)
    similarity = normalized @ normalized.T
    rank_k = max(1, min(int(rank_k), len(logits)))
    detached = similarity.detach()
    detached = detached - torch.diagflat(torch.diagonal(detached))
    cutoff = torch.topk(detached, rank_k, dim=1).values.min(1).values
    mask = (detached > cutoff[:, None]).float().fill_diagonal_(1)
    stable_similarity = similarity + torch.finfo(torch.float32).tiny * torch.ones_like(similarity)
    neighbors = (stable_similarity / stable_similarity.detach()) * mask
    relation = neighbors @ neighbors.T / rank_k
    mask = (relation > threshold).float()
    relation = relation + torch.finfo(torch.float32).tiny * torch.ones_like(relation)
    hard_relation = (relation / relation.detach()) * mask
    return similarity, relation, hard_relation


def relation_loss(fusion_relation, view_relations):
    losses = []
    for similarity, relation, _ in view_relations:
        first = F.cross_entropy(fusion_relation.detach(), similarity, reduction='sum')
        second = F.cross_entropy(relation, fusion_relation, reduction='sum')
        losses.append((0.0001 * first + 0.01 * second) / len(similarity))
    return torch.stack(losses).mean()
