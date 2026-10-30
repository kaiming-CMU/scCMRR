import math
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
import torch
from torch import nn
from torch.nn import functional as F

@torch.no_grad()
def semi_relaxed_transport(logits, prior, epsilon=0.1, tau=1.0, temperature=1.0, max_iterations=500, tolerance=1e-07):
    if not math.isfinite(epsilon) or epsilon <= 0 or (not math.isfinite(tau)) or (tau < 0) or (not math.isfinite(temperature)) or (temperature <= 0) or (not math.isfinite(tolerance)) or (tolerance <= 0) or (not isinstance(max_iterations, int)) or (max_iterations < 1):
        raise ValueError('Invalid OT regularization, temperature, tolerance or iterations.')
    if logits.ndim != 2 or min(logits.shape) < 1 or (not torch.isfinite(logits).all()):
        raise ValueError('OT logits must be a finite nonempty N x K matrix.')
    if not logits.is_floating_point():
        raise ValueError('OT logits must be floating point.')
    (n, k) = logits.shape
    pi = torch.as_tensor(prior, dtype=torch.float64, device=logits.device)
    if pi.shape != (k,) or not torch.isfinite(pi).all() or torch.any(pi <= 0):
        raise ValueError('OT prior must have K finite, strictly positive entries.')
    pi = pi / pi.sum()
    log_pi = pi.log()
    log_kernel = F.log_softmax(logits.detach().double() / temperature, dim=1) / epsilon
    if not torch.isfinite(log_kernel).all():
        raise ValueError('OT logits/temperature/epsilon exceed numerical range.')
    log_v = torch.zeros_like(pi)
    exponent = tau / (tau + epsilon)
    converged = False
    for iteration in range(1, max_iterations + 1):
        log_u = -math.log(n) - torch.logsumexp(log_kernel + log_v[None, :], dim=1)
        updated = exponent * (log_pi - torch.logsumexp(log_kernel + log_u[:, None], dim=0))
        updated = updated - updated.mean()
        error = (updated - log_v).abs().max().item()
        log_v = updated
        if error <= tolerance:
            converged = True
            break
    log_p = F.log_softmax(log_kernel + log_v[None, :], dim=1)
    p = log_p.exp()
    log_mass = torch.logsumexp(log_p, dim=0) - math.log(n)
    mass = log_mass.exp()
    residual = epsilon * log_v + tau * (log_mass - log_pi)
    residual = (residual - residual.mean()).abs().max().item()
    diagnostics = dict(iterations=iteration, converged=converged, fixed_point_error=error, stationarity_residual=residual, row_error=(p.sum(1) - 1).abs().max().item(), prior=pi.cpu().tolist(), target_mass=mass.cpu().tolist(), prior_kl=(mass * (log_mass - log_pi)).sum().item(), target_entropy=(-(p * log_p).sum(1)).mean().item(), effective_clusters=(-(mass * log_mass).sum()).exp().item(), hard_occupied_clusters=int(p.argmax(1).unique().numel()))
    return (p.to(logits.dtype), diagnostics)

class SemiRelaxedOT(nn.Module):

    def __init__(self, n_clusters, epsilon=0.1, tau=1.0, temperature=1.0, max_iterations=500, tolerance=1e-07, warmup_epochs=20, ramp_epochs=20, prior_smoothing=0.001):
        super().__init__()
        if n_clusters < 1:
            raise ValueError('n_clusters must be positive.')
        if not isinstance(warmup_epochs, int) or warmup_epochs < 0 or (not isinstance(ramp_epochs, int)) or (ramp_epochs < 0):
            raise ValueError('OT warm-up and ramp must be nonnegative integers.')
        if not 0 < prior_smoothing < 1:
            raise ValueError('prior_smoothing must lie strictly between zero and one.')
        semi_relaxed_transport(torch.zeros(1, 1), torch.ones(1), epsilon, tau, temperature, max_iterations, tolerance)
        (self.epsilon, self.tau, self.temperature) = (epsilon, tau, temperature)
        (self.max_iterations, self.tolerance) = (max_iterations, tolerance)
        (self.warmup_epochs, self.ramp_epochs) = (warmup_epochs, ramp_epochs)
        self.prior_smoothing = prior_smoothing
        self.register_buffer('prior', torch.full((n_clusters,), 1.0 / n_clusters))
        self.register_buffer('prior_initialized', torch.tensor(False))
        self.register_buffer('prior_alignment', torch.arange(n_clusters))
        self.register_buffer('initial_cluster_counts', torch.zeros(n_clusters, dtype=torch.long))
        self.register_buffer('current_epoch', torch.tensor(0, dtype=torch.long))
        self.last_diagnostics = {}

    def set_epoch(self, epoch):
        if not isinstance(epoch, int) or epoch < 0:
            raise ValueError('epoch must be a nonnegative integer.')
        self.current_epoch.fill_(epoch)

    @torch.no_grad()
    def initialize_prior(self, features, teacher_logits):
        if bool(self.prior_initialized):
            return
        k = self.prior.numel()
        if features.ndim != 2 or len(features) < k or teacher_logits.shape != (len(features), k) or (not torch.isfinite(features).all()) or (not torch.isfinite(teacher_logits).all()):
            raise ValueError('Invalid features/logits for OT prior initialization.')
        assignments = KMeans(n_clusters=k, n_init=10).fit_predict(features.detach().float().cpu().numpy())
        probabilities = F.softmax(teacher_logits.detach().double() / self.temperature, dim=1).cpu().numpy()
        counts = np.bincount(assignments, minlength=k)
        agreement = np.zeros((k, k), dtype=np.float64)
        for cluster in range(k):
            if counts[cluster]:
                agreement[cluster] = probabilities[assignments == cluster].mean(axis=0)
        (rows, columns) = linear_sum_assignment(-agreement)
        aligned = np.zeros(k, dtype=np.float64)
        aligned[columns] = counts[rows] / len(features)
        aligned = (1 - self.prior_smoothing) * aligned + self.prior_smoothing / k
        self.prior.copy_(torch.as_tensor(aligned, device=self.prior.device, dtype=self.prior.dtype))
        self.initial_cluster_counts.copy_(torch.as_tensor(counts, device=self.prior.device))
        self.prior_alignment[torch.as_tensor(rows, device=self.prior.device)] = torch.as_tensor(columns, device=self.prior.device)
        self.prior_initialized.fill_(True)

    def forward(self, teacher_logits, student_logits, features):
        if not student_logits or any((s.shape != teacher_logits.shape for s in student_logits)):
            raise ValueError('All student logits must match the teacher N x K shape.')
        epoch = int(self.current_epoch)
        weight = min(1.0, (epoch - self.warmup_epochs) / max(1, self.ramp_epochs))
        if epoch <= self.warmup_epochs:
            self.last_diagnostics = dict(epoch=epoch, phase='warmup', weight=0.0, prior_initialized=bool(self.prior_initialized))
            return sum((s.sum() * 0 for s in student_logits))
        if not bool(self.prior_initialized):
            if not self.training:
                raise RuntimeError('OT prior not initialized; initialize during training after warm-up.')
            self.initialize_prior(features, teacher_logits)
        with torch.no_grad():
            (target, diagnostics) = semi_relaxed_transport(teacher_logits, self.prior, self.epsilon, self.tau, self.temperature, self.max_iterations, self.tolerance)
        if not diagnostics['converged']:
            raise RuntimeError(f'OT did not converge: {diagnostics}; increase max_iterations or epsilon.')
        loss = torch.stack([-(target * F.log_softmax(s / self.temperature, dim=1)).sum(1).mean() for s in student_logits]).mean()
        self.last_diagnostics = dict(diagnostics, epoch=epoch, phase='active', weight=weight, raw_loss=float(loss.detach()), prior_initialized=True, initial_cluster_counts=self.initial_cluster_counts.cpu().tolist(), prior_alignment=self.prior_alignment.cpu().tolist())
        return weight * loss
