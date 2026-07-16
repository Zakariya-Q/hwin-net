"""
HWIN-Net: Metrics and Evaluation

Mathematical Purpose
--------------------
Implements evaluation metrics for HWIN-Net:
- Prediction accuracy metrics
- Routing accuracy (identifiability gate)
- Uncertainty calibration
- Equivariance quality
- Retraction quality (idempotence)

Theory Traceability
-------------------
- Theorems 3, 4, 5 (SIS properties)
- Lemmas 5, 7, 8
- Axioms A4, A5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings


class NMSE:
    """Normalized Mean Squared Error."""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.sum_sq_error = 0.0
        self.sum_sq_target = 0.0
        self.n = 0
    
    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        preds = preds.detach().cpu().numpy().flatten()
        targets = targets.detach().cpu().numpy().flatten()
        self.sum_sq_error += np.sum((preds - targets) ** 2)
        self.sum_sq_target += np.sum(targets ** 2)
        self.n += len(targets)
    
    def compute(self):
        if self.sum_sq_target == 0:
            return float("nan")
        return float(self.sum_sq_error / self.sum_sq_target)


class NLL:
    """Negative Log-Likelihood for Gaussian predictions."""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.sum_nll = 0.0
        self.n = 0
    
    def update(self, preds: torch.Tensor, targets: torch.Tensor, variances: torch.Tensor):
        preds = preds.detach().cpu().numpy().flatten()
        targets = targets.detach().cpu().numpy().flatten()
        variances = variances.detach().cpu().numpy().flatten()
        
        # Gaussian NLL: 0.5 * (log(2*pi*var) + (y - mu)^2 / var)
        nll = 0.5 * (np.log(2 * np.pi * variances) + (targets - preds) ** 2 / variances)
        self.sum_nll += np.sum(nll)
        self.n += len(targets)
    
    def compute(self):
        if self.n == 0:
            return float("nan")
        return float(self.sum_nll / self.n)


class ECE:
    """Expected Calibration Error for regression."""
    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
        self.reset()
    
    def reset(self):
        self.bin_errors = np.zeros(self.n_bins)
        self.bin_confidences = np.zeros(self.n_bins)
        self.bin_counts = np.zeros(self.n_bins, dtype=int)
    
    def update(self, preds: torch.Tensor, targets: torch.Tensor, uncertainties: torch.Tensor):
        preds = preds.detach().cpu().numpy().flatten()
        targets = targets.detach().cpu().numpy().flatten()
        uncertainties = uncertainties.detach().cpu().numpy().flatten()
        
        abs_error = np.abs(preds - targets)
        pred_std = np.sqrt(uncertainties + 1e-8)
        
        for i in range(self.n_bins):
            # Quantile-based binning
            lower = i / self.n_bins
            upper = (i + 1) / self.n_bins
            lower_val = np.quantile(pred_std, lower)
            upper_val = np.quantile(pred_std, upper)
            
            mask = (pred_std >= lower_val) & (pred_std < upper_val)
            if np.sum(mask) > 0:
                bin_error = np.mean(abs_error[mask])
                bin_conf = np.mean(pred_std[mask])
                self.bin_errors[i] += bin_error * np.sum(mask)
                self.bin_confidences[i] += bin_conf * np.sum(mask)
                self.bin_counts[i] += np.sum(mask)
    
    def compute(self):
        ece = 0.0
        total = np.sum(self.bin_counts)
        if total == 0:
            return float("nan")
        for i in range(self.n_bins):
            if self.bin_counts[i] > 0:
                avg_error = self.bin_errors[i] / self.bin_counts[i]
                avg_conf = self.bin_confidences[i] / self.bin_counts[i]
                ece += abs(avg_error - avg_conf) * (self.bin_counts[i] / total)
        return float(ece)


class MCE:
    """Maximum Calibration Error."""
    def __init__(self, n_bins: int = 10):
        self.ece = ECE(n_bins)
    
    def reset(self):
        self.ece.reset()
    
    def update(self, preds, targets, uncertainties):
        self.ece.update(preds, targets, uncertainties)
    
    def compute(self):
        self.ece.compute()  # populates bins
        max_err = 0.0
        for i in range(self.ece.n_bins):
            if self.ece.bin_counts[i] > 0:
                avg_error = self.ece.bin_errors[i] / self.ece.bin_counts[i]
                avg_conf = self.ece.bin_confidences[i] / self.ece.bin_counts[i]
                max_err = max(max_err, abs(avg_error - avg_conf))
        return float(max_err)


class TACE:
    """Thresholded Adaptive Calibration Error."""
    def __init__(self, threshold: float = 0.1):
        self.threshold = threshold
        self.errors = []
        self.confidences = []
    
    def reset(self):
        self.errors = []
        self.confidences = []
    
    def update(self, preds, targets, uncertainties):
        preds = preds.detach().cpu().numpy().flatten()
        targets = targets.detach().cpu().numpy().flatten()
        uncertainties = uncertainties.detach().cpu().numpy().flatten()
        
        abs_error = np.abs(preds - targets)
        pred_std = np.sqrt(uncertainties + 1e-8)
        
        mask = pred_std > self.threshold
        if np.any(mask):
            self.errors.extend(abs_error[mask])
            self.confidences.extend(pred_std[mask])
    
    def compute(self):
        if len(self.errors) == 0:
            return float("nan")
        return float(np.mean(np.abs(np.array(self.errors) - np.array(self.confidences))))


class Sharpness:
    """Sharpness of prediction intervals."""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.widths = []
    
    def update(self, lower: torch.Tensor, upper: torch.Tensor):
        self.widths.extend((upper - lower).detach().cpu().numpy().flatten())
    
    def compute(self):
        if len(self.widths) == 0:
            return float("nan")
        return float(np.mean(self.widths))


class RoutingAccuracy:
    """Accuracy of identifiability gate routing."""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.correct = 0
        self.total = 0
    
    def update(self, routed: torch.Tensor, true_routed: torch.Tensor):
        routed = routed.detach().cpu().numpy()
        true_routed = true_routed.detach().cpu().numpy()
        self.correct += np.sum(routed == true_routed)
        self.total += len(routed)
    
    def compute(self):
        if self.total == 0:
            return float("nan")
        return float(self.correct / self.total)


class ThresholdAccuracy:
    """Accuracy of threshold-based routing."""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.correct = 0
        self.total = 0
    
    def update(self, cardinalities: torch.Tensor, r0_vals: torch.Tensor, routed: torch.Tensor):
        card = cardinalities.detach().cpu().numpy()
        r0 = r0_vals.detach().cpu().numpy()
        routed = routed.detach().cpu().numpy()
        
        true_routed = (card < r0).astype(float)
        self.correct += np.sum(routed == true_routed)
        self.total += len(routed)
    
    def compute(self):
        if self.total == 0:
            return float("nan")
        return float(self.correct / self.total)


class RecoveryError:
    """Recovery module error (mu_hat vs mu_true)."""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.sum_sq = 0.0
        self.n = 0
    
    def update(self, mu_hat: torch.Tensor, mu_true: torch.Tensor):
        diff = (mu_hat - mu_true).detach().cpu().numpy()
        self.sum_sq += np.sum(diff ** 2)
        self.n += diff.shape[0]
    
    def compute(self):
        if self.n == 0:
            return float("nan")
        return float(np.sqrt(self.sum_sq / self.n))


class EquivarianceError:
    """Equivariance error for recovery module."""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.sum_fro = 0.0
        self.n = 0
    
    def update(self, R_ab: torch.Tensor, R_bc: torch.Tensor, R_ac: torch.Tensor):
        # Check R_bc @ R_ac ? R_ab
        composed = R_bc @ R_ac
        fro = torch.norm(composed - R_ab, p="fro").item()
        self.sum_fro += fro
        self.n += 1
    
    def compute(self):
        if self.n == 0:
            return float("nan")
        return float(self.sum_fro / self.n)


class ManifoldIdempotence:
    """Idempotence error for manifold retraction."""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.errs = []
    
    def update(self, mu_final: torch.Tensor, retraction_module: nn.Module):
        with torch.no_grad():
            mu_double = retraction_module(mu_final)
            err = (mu_double - mu_final).norm(dim=-1).mean().item()
            self.errs.append(err)
    
    def compute(self):
        if len(self.errs) == 0:
            return float("nan")
        return float(np.mean(self.errs))


class ManifoldFixing:
    """Manifold fixing property: ||rho(mu) - mu|| for mu on manifold."""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.errs = []
    
    def update(self, mu_on_manifold: torch.Tensor, retraction_module: nn.Module):
        with torch.no_grad():
            mu_projected = retraction_module(mu_on_manifold)
            err = (mu_projected - mu_on_manifold).norm(dim=-1).mean().item()
            self.errs.append(err)
    
    def compute(self):
        if len(self.errs) == 0:
            return float("nan")
        return float(np.mean(self.errs))


class MINE:
    """Mutual Information Neural Estimation."""
    def __init__(self, x_dim: int, y_dim: int, hidden: int = 64):
        self.net = nn.Sequential(
            nn.Linear(x_dim + y_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, y], dim=-1))
    
    def compute_mi(self, x: torch.Tensor, y: torch.Tensor, y_shuffled: Optional[torch.Tensor] = None) -> float:
        if y_shuffled is None:
            idx = torch.randperm(y.shape[0])
            y_shuffled = y[idx]
        
        with torch.no_grad():
            t = self.forward(x, y)
            t_shuffled = self.forward(x, y_shuffled)
            
            mi = t.mean() - torch.log(torch.exp(t_shuffled).mean() + 1e-8)
            return mi.item()


class UncertaintyDecomposition:
    """Decompose total uncertainty into aleatoric, epistemic, non-identifiability."""
    def __init__(self):
        pass
    
    def decompose(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {
            "aleatoric": outputs.get("sigma2_aleat", torch.zeros_like(outputs.get("sigma2_total"))),
            "epistemic": outputs.get("sigma2_epi", torch.zeros_like(outputs.get("sigma2_total"))),
            "non_identifiable": outputs.get("sigma2_nonid", torch.zeros_like(outputs.get("sigma2_total"))),
            "total": outputs.get("sigma2_total", torch.zeros_like(outputs.get("sigma2_aleat")))
        }


def compute_prediction_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    uncertainties: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Compute standard prediction metrics."""
    preds = preds.detach().cpu().numpy().flatten()
    targets = targets.detach().cpu().numpy().flatten()
    
    if len(preds) == 0:
        return {}
    
    metrics = {
        "mse": float(mean_squared_error(targets, preds)),
        "rmse": float(np.sqrt(mean_squared_error(targets, preds))),
        "mae": float(mean_absolute_error(targets, preds)),
        "r2": float(r2_score(targets, preds)) if len(preds) > 1 else 0.0,
    }
    
    # Correlation
    if len(preds) > 1:
        corr = np.corrcoef(preds, targets)[0, 1]
        metrics["correlation"] = float(corr) if not np.isnan(corr) else 0.0
    
    # Uncertainty-aware metrics
    if uncertainties is not None:
        uncerts = uncertainties.detach().cpu().numpy().flatten()
        abs_error = np.abs(preds - targets)
        
        # Uncertainty calibration
        within_1sigma = (abs_error <= np.sqrt(uncerts + 1e-8)).mean()
        metrics["within_1sigma"] = float(within_1sigma)
        within_2sigma = (abs_error <= 2 * np.sqrt(uncerts + 1e-8)).mean()
        metrics["within_2sigma"] = float(within_2sigma)
        
        # Uncertainty-error correlation
        if len(uncerts) > 1:
            unc_corr = np.corrcoef(abs_error, np.sqrt(uncerts + 1e-8))[0, 1]
            metrics["uncertainty_error_correlation"] = float(unc_corr) if not np.isnan(unc_corr) else 0.0
    
    return metrics


def compute_routing_metrics(
    routed: torch.Tensor,
    true_routed: Optional[torch.Tensor] = None,
    r0_vals: Optional[torch.Tensor] = None,
    cardinalities: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Compute identifiability gate routing metrics."""
    routed = routed.detach().cpu().numpy()
    
    metrics = {
        "routed_fraction": float(routed.mean()),
        "n_routed": int(routed.sum()),
        "n_not_routed": int((1 - routed).sum()),
    }
    
    if true_routed is not None:
        true_routed = true_routed.detach().cpu().numpy()
        metrics["routing_accuracy"] = float((routed == true_routed).mean())
        metrics["routing_precision"] = float(
            (routed & true_routed).sum() / max(1, routed.sum())
        )
        metrics["routing_recall"] = float(
            (routed & true_routed).sum() / max(1, true_routed.sum())
        )
    
    if r0_vals is not None and cardinalities is not None:
        r0 = r0_vals.detach().cpu().numpy()
        card = cardinalities.detach().cpu().numpy()
        
        expected_routed = (card < r0).astype(float)
        metrics["routed_vs_theory"] = float((routed == expected_routed).mean())
        
        false_positive = ((routed == 1) & (card >= r0)).mean()
        false_negative = ((routed == 0) & (card < r0)).mean()
        metrics["routing_false_positive"] = float(false_positive)
        metrics["routing_false_negative"] = float(false_negative)
    
    return metrics


def compute_uncertainty_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    aleatoric: Optional[torch.Tensor] = None,
    epistemic: Optional[torch.Tensor] = None,
    non_id: Optional[torch.Tensor] = None,
    total: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Compute uncertainty calibration metrics."""
    preds = preds.detach().cpu().numpy().flatten()
    targets = targets.detach().cpu().numpy().flatten()
    abs_error = np.abs(preds - targets)
    
    metrics = {}
    
    if total is not None:
        total_u = total.detach().cpu().numpy().flatten()
        metrics["mean_total_uncertainty"] = float(np.sqrt(total_u + 1e-8).mean())
        metrics.update(_calibration_metrics(abs_error, np.sqrt(total_u + 1e-8)))
    
    if aleatoric is not None:
        alea = aleatoric.detach().cpu().numpy().flatten()
        metrics["mean_aleatoric"] = float(np.sqrt(alea + 1e-8).mean())
        metrics.update({f"aleatoric_{k}": v for k, v in _calibration_metrics(abs_error, np.sqrt(alea + 1e-8)).items()})
    
    if epistemic is not None:
        epi = epistemic.detach().cpu().numpy().flatten()
        metrics["mean_epistemic"] = float(np.sqrt(epi + 1e-8).mean())
        metrics.update({f"epistemic_{k}": v for k, v in _calibration_metrics(abs_error, np.sqrt(epi + 1e-8)).items()})
    
    if non_id is not None:
        nid = non_id.detach().cpu().numpy().flatten()
        metrics["mean_non_id"] = float(np.sqrt(nid + 1e-8).mean())
    
    return metrics


def _calibration_metrics(abs_error: np.ndarray, predicted_std: np.ndarray) -> Dict[str, float]:
    """Internal calibration metrics."""
    if len(abs_error) == 0:
        return {}
    
    n_bins = 10
    idx = np.argsort(predicted_std)
    abs_error = abs_error[idx]
    predicted_std = predicted_std[idx]
    
    bin_size = len(abs_error) // n_bins
    ece = 0.0
    for i in range(n_bins):
        start = i * bin_size
        end = (i + 1) * bin_size if i < n_bins - 1 else len(abs_error)
        if end <= start:
            continue
        
        bin_error = abs_error[start:end]
        bin_std = predicted_std[start:end]
        
        within = (bin_error <= bin_std).mean()
        conf = within
        acc = within
        ece += abs(acc - conf) * (end - start) / len(abs_error)
    
    return {
        "ece": float(ece),
        "error_std_correlation": float(np.corrcoef(abs_error, predicted_std)[0, 1]) if len(abs_error) > 1 else 0.0,
    }


def compute_equivariance_metrics(
    intertwiners: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    """Compute equivariance quality metrics."""
    metrics = {}
    
    for name, R in intertwiners.items():
        if R is None or isinstance(R, nn.Identity):
            continue
        
        if isinstance(R, torch.Tensor):
            W = R
        elif hasattr(R, "weight"):
            W = R.weight
        elif hasattr(R, "transform") and hasattr(R.transform, "weight"):
            W = R.transform.weight
        else:
            continue
        
        W = W.detach().cpu()
        
        diff = W - torch.eye(W.shape[0])
        metrics[f"{name}_frobenius_from_I"] = float(torch.norm(diff, p="fro").item())
        metrics[f"{name}_spectral_from_I"] = float(torch.linalg.svd(diff)[0].max().item())
        
        WTW = W.T @ W
        metrics[f"{name}_orthogonality"] = float(torch.norm(WTW - torch.eye(W.shape[0]), p="fro").item())
        
        metrics[f"{name}_det"] = float(torch.det(W).item())
    
    return metrics


def compute_retraction_metrics(
    mu_hat: torch.Tensor,
    mu_final: torch.Tensor,
    retraction_module: Optional[nn.Module] = None,
) -> Dict[str, float]:
    """Compute retraction quality: idempotence, manifold fixing."""
    metrics = {}
    
    displacement = (mu_final - mu_hat).norm(dim=-1)
    metrics["mean_displacement"] = float(displacement.mean().item())
    metrics["max_displacement"] = float(displacement.max().item())
    
    if retraction_module is not None:
        with torch.no_grad():
            mu_double = retraction_module(mu_final)
            idempotence_error = (mu_double - mu_final).norm(dim=-1)
            metrics["idempotence_error_mean"] = float(idempotence_error.mean().item())
            metrics["idempotence_error_max"] = float(idempotence_error.max().item())
    
    return metrics


def compute_complexity_metrics(
    mu_final: torch.Tensor,
) -> Dict[str, float]:
    """Compute latent complexity metrics."""
    mu_final = mu_final.detach()
    
    return {
        "latent_l2_mean": float(mu_final.pow(2).mean().item()),
        "latent_l2_max": float(mu_final.pow(2).max().item()),
        "latent_l1_mean": float(mu_final.abs().mean().item()),
        "latent_norm_std": float(mu_final.norm(dim=-1).std().item()),
        "latent_dims_active": int((mu_final.abs() > 1e-4).sum(dim=-1).float().mean().item()),
    }


def compute_all_metrics(
    outputs: Dict[str, torch.Tensor],
    targets: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Compute all available metrics from model outputs."""
    metrics = {}
    
    if targets is not None:
        pred_metrics = compute_prediction_metrics(
            outputs.get("q_out", outputs.get("q_hat")),
            targets,
            outputs.get("sigma2_total"),
        )
        metrics.update({f"pred_{k}": v for k, v in pred_metrics.items()})
    
    if "routed" in outputs:
        routing_metrics = compute_routing_metrics(
            outputs["routed"],
            r0_vals=outputs.get("r0_vals"),
            cardinalities=outputs.get("M_O").sum(dim=1) if "M_O" in outputs else None,
        )
        metrics.update(routing_metrics)
    
    if "sigma2_total" in outputs:
        uncert_metrics = compute_uncertainty_metrics(
            outputs.get("q_out", outputs.get("q_hat")),
            targets if targets is not None else torch.zeros_like(outputs.get("q_out", outputs.get("q_hat"))),
            outputs.get("sigma2_aleat"),
            outputs.get("sigma2_epi"),
            outputs.get("sigma2_nonid"),
            outputs.get("sigma2_total"),
        )
        metrics.update(uncert_metrics)
    
    if "mu_final" in outputs:
        comp_metrics = compute_complexity_metrics(outputs["mu_final"])
        metrics.update(comp_metrics)
    
    if "mu_hat" in outputs and "mu_final" in outputs:
        ret_metrics = compute_retraction_metrics(outputs["mu_hat"], outputs["mu_final"])
        metrics.update(ret_metrics)
    
    return metrics
