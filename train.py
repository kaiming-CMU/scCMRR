import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans

from dataset import MMDataset, get_config
from metrics import clustering_metrics
from model.network import scCMRR


def train(config):
    if not isinstance(config['epochs'], int) or config['epochs'] < 1:
        raise ValueError('epochs must be a positive integer.')
    requested = config['device']
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if requested == 'auto' else requested)
    data = MMDataset(config['data_path'], config['modality_keys'], config['label_key'],
                     config['modality_types'], config['n_neighbors'])
    model = scCMRR([v.shape[1] for v in data.x], config['hidden_dim'], len(data),
                   data.n_classes, config['rank_k'], config['theta'], config['ot']).to(device)
    x = [torch.as_tensor(v, device=device) for v in data.x]
    adjacency = [torch.as_tensor(a, device=device) for a in data.adj]
    fusion_x = data.f_x.to(device)
    fusion_adjacency = torch.as_tensor(data.f_adj, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    output = Path(config['result_path'])
    output.mkdir(parents=True, exist_ok=True)
    best = None
    history = []
    for epoch in range(1, config['epochs'] + 1):
        model.train()
        model.set_epoch(epoch)
        rec, crr, sot = model(x, adjacency, fusion_x, fusion_adjacency)
        loss = rec + config['lambda_1'] * crr + config['lambda_2'] * sot
        if not torch.isfinite(loss):
            raise RuntimeError(f'Non-finite loss at epoch {epoch}.')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise RuntimeError(f'Non-finite gradients at epoch {epoch}.')
        optimizer.step()
        losses = {'REC': float(rec.detach()), 'CRR': float(crr.detach()), 'SOT': float(sot.detach()), 'loss': float(loss.detach())}
        del rec, crr, sot, loss
        model.eval()
        with torch.no_grad():
            fusion, combined = model.embeddings(x, adjacency, fusion_x, fusion_adjacency)
            for branch, features in [('fusion', fusion), ('combined', combined)]:
                embedding = features.cpu().numpy()
                predictions = KMeans(n_clusters=data.n_classes, n_init=10).fit_predict(embedding)
                scores = clustering_metrics(data.y, predictions)
                history.append({'epoch': epoch, 'branch': branch, **losses, **scores})
                if best is None or scores['ACC'] > best['ACC']:
                    best = {'dataset': config['dataset_name'], 'epoch': epoch, 'branch': branch,
                            'n_cells': len(data), 'n_clusters': data.n_classes, **scores}
                    best_embedding = embedding.copy()
                    best_predictions = predictions.copy()
                    if config['save_model']:
                        best_weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f'Epoch {epoch}/{config["epochs"]} loss={losses["loss"]:.6f} best_ACC={best["ACC"]:.4f}', flush=True)
    np.savez_compressed(output / 'clustering.npz', embedding=best_embedding,
                        predictions=best_predictions, labels=data.y, label_names=data.label_names)
    (output / 'metrics.json').write_text(json.dumps(best, indent=2), encoding='utf-8')
    (output / 'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
    if config['save_model']:
        saved_config = deepcopy(config)
        saved_config['data_path'] = Path(config['data_path']).name
        saved_config['result_path'] = '.'
        torch.save({'state_dict': best_weights, 'config': saved_config, 'metrics': best}, output / 'model.pt')
    print(' '.join(f'{key}={best[key]:.4f}' for key in ('ACC', 'NMI', 'ARI', 'AMI')), flush=True)
    return best


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--device')
    parser.add_argument('--output')
    args = parser.parse_args()
    config = get_config(args.config)
    for key in ('epochs', 'device'):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    if args.output is not None:
        config['result_path'] = str(Path(args.output).resolve())
    train(config)
