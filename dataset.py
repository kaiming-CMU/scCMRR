import json
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler, normalize


def get_config(config_path=None):
    path = Path(config_path) if config_path else Path(__file__).with_name('config.json')
    path = path.resolve()
    with path.open(encoding='utf-8') as handle:
        config = json.load(handle)
    for key in ('data_path', 'result_path'):
        value = Path(config[key])
        config[key] = str(value if value.is_absolute() else path.parent / value)
    return config


def read_labels(node):
    if isinstance(node, h5py.Group):
        codes = node['codes'][:]
        categories = node['categories'][:]
        if codes.ndim != 1 or np.any(codes < 0) or np.any(codes >= len(categories)):
            raise ValueError('Invalid or missing categorical labels.')
        values = categories[codes]
    else:
        values = node[:].reshape(-1)
    if values.dtype.kind in 'fiu' and not np.isfinite(values).all():
        raise ValueError('Labels must be finite.')
    return values.astype(str)


def read_matrix(node):
    if isinstance(node, h5py.Dataset):
        if node.ndim != 2:
            raise ValueError('Each modality must be a cell-by-feature matrix.')
        return sparse.csr_matrix(node.astype('float32')[:])
    encoding = node.attrs.get('encoding-type', node.attrs.get('format', 'csr'))
    if isinstance(encoding, bytes):
        encoding = encoding.decode()
    if not str(encoding).startswith(('csr', 'csc')):
        raise ValueError('Sparse modalities must use CSR or CSC storage.')
    matrix = sparse.csc_matrix if str(encoding).startswith('csc') else sparse.csr_matrix
    return matrix((node['data'][:].astype(np.float32), node['indices'][:],
                   node['indptr'][:]), shape=tuple(node.attrs['shape'])).tocsr()


def preprocess(x, kind):
    if kind not in ('rna', 'logged_rna', 'atac', 'protein'):
        raise ValueError('Unknown modality type.')
    if not np.isfinite(x.data).all() or np.any(x.data < 0):
        raise ValueError('Modality values must be finite and nonnegative.')
    if kind == 'rna':
        x = normalize(x, norm='l1', axis=1) * 10000
        x.data = np.log1p(x.data)
    elif kind == 'protein':
        x.data = np.log1p(x.data)
    elif kind == 'atac':
        x = normalize(x, norm='l1', axis=1)
        idf = np.log1p(x.shape[0] / (1 + np.asarray((x > 0).sum(axis=0)).ravel()))
        x = x.multiply(idf).tocsr()
    mean = np.asarray(x.mean(axis=0)).ravel()
    variance = np.maximum(np.asarray(x.multiply(x).mean(axis=0)).ravel() - mean ** 2, 0)
    available = np.flatnonzero(variance > 0)
    limit = 10000 if kind == 'atac' else 2000
    order = np.argsort(-variance[available], kind='stable')[:limit]
    selected = np.sort(available[order])
    if not len(selected):
        raise ValueError('Modality has no variable features.')
    x = x[:, selected]
    if x.shape[1] > 50:
        x = TruncatedSVD(n_components=min(50, x.shape[0] - 1, x.shape[1] - 1),
                         n_iter=7).fit_transform(x)
    else:
        x = x.toarray()
    x = StandardScaler().fit_transform(x).astype(np.float32)
    if not np.isfinite(x).all():
        raise ValueError('Non-finite preprocessed features.')
    return x


def cell_graph(x, n_neighbors):
    graph = kneighbors_graph(x, n_neighbors=min(n_neighbors, len(x) - 1),
                            mode='connectivity', include_self=False, n_jobs=1)
    return graph.maximum(graph.T).toarray().astype(np.float32)


class MMDataset:
    def __init__(self, data_path, modality_keys, label_key, modality_types, n_neighbors=20):
        if len(modality_keys) < 2 or len(set(modality_keys)) != len(modality_keys):
            raise ValueError('Specify at least two distinct modality keys.')
        if len(modality_keys) != len(modality_types):
            raise ValueError('Each modality key must have a corresponding modality type.')
        if not isinstance(n_neighbors, int) or n_neighbors < 1:
            raise ValueError('n_neighbors must be a positive integer.')
        self.x = []
        with h5py.File(data_path, 'r') as data:
            labels = read_labels(data[label_key])
            if len(labels) < 2:
                raise ValueError('At least two cells are required.')
            self.label_names, self.y = np.unique(labels, return_inverse=True)
            for key, kind in zip(modality_keys, modality_types):
                matrix = read_matrix(data[key])
                if matrix.shape[0] != len(labels) or matrix.shape[1] == 0:
                    raise ValueError('All modalities must contain aligned cells and nonempty features.')
                self.x.append(preprocess(matrix, kind))
        self.n_classes = len(self.label_names)
        self.adj = [cell_graph(x, n_neighbors) for x in self.x]
        self.f_x = torch.from_numpy(np.concatenate(self.x, axis=1))
        self.f_adj = cell_graph(self.f_x.numpy(), n_neighbors)
        self.modality_keys = tuple(modality_keys)

    def __len__(self):
        return len(self.y)
