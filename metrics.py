import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics.cluster import contingency_matrix


def clustering_metrics(labels, predictions):
    matrix = contingency_matrix(labels, predictions)
    rows, columns = linear_sum_assignment(-matrix)
    return {
        'ACC': float(matrix[rows, columns].sum() / matrix.sum()),
        'NMI': float(normalized_mutual_info_score(labels, predictions)),
        'ARI': float(adjusted_rand_score(labels, predictions)),
        'AMI': float(adjusted_mutual_info_score(labels, predictions)),
    }
