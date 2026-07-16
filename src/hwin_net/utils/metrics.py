"""
HWIN-Net: Metrics Package
=========================

Evaluation metrics for HWIN-Net experiments:
- Regression metrics: NMSE, NLL
- Calibration metrics: ECE, MCE, TACE, Sharpness
- Gate/Router metrics: RoutingAccuracy, ThresholdAccuracy
- Recovery metrics: RecoveryError
- Equivariance metrics: EquivarianceError
- Manifold metrics: ManifoldIdempotence, ManifoldFixing
- MI estimation: MINE
- Uncertainty decomposition: UncertaintyDecomposition
- Composite: compute_all_metrics
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Any, Tuple
import math


def NMSE(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Normalized Mean Squared Error."""
    mask = ~torch.isnan(y_true)
    if not mask.any():
        return torch.tensor(float("nan"), device=y_pred.device)
    y_pred = y_pred[mask]
    y_true = y_true[mask]
    mse = F.mse_loss(y_pred, y_true)
    var = y_true.var(unbiased=False)
    return mse / (var + 1e-8)


def NLL(y_pred: torch.Tensor, y_true: torch.Tensor, sigma2: torch.Tensor) -> torch.Tensor:
    """Gaussian Negative Log-Likelihood."""
    mask = ~torch.isnan(y_true)
    if not mask.any():
        return torch.tensor(float("nan"), device=y_pred.device)
    y_pred = y_pred[mask]
    y_true = y_true[mask]
    sigma2 = sigma2[mask]
    eps = 1e-6
    sigma2 = sigma2.clamp(min=eps)
    return 0.5 * (torch.log(2 * math.pi * sigma2) + (y_pred - y_true) ** 2 / sigma2).mean()


def ECE(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    sigma2: torch.Tensor,
    n_bins: int = 15,
) -> torch.Tensor:
    """Expected Calibration Error for regression."""
    mask = ~torch.isnan(y_true)
    if not mask.any():
        return torch.tensor(float("nan"), device=y_pred.device)
    y_pred = y_pred[mask]
    y_true = y_true[mask]
    sigma2 = sigma2[mask]
    
    sigma = torch.sqrt(sigma2.clamp(min=1e-8))
    z = (y_true - y_pred) / sigma
    
    # Use standard normal CDF for confidence intervals
    from torch.distributions import Normal
    normal = Normal(0, 1)
    
    ece = torch.tensor(0.0, device=y_pred.device)
    for i in range(n_bins):
        conf_low = i / n_bins
        conf_high = (i + 1) / n_bins
        lower = normal.icdf(torch.tensor(conf_low, device=y_pred.device))
        upper = normal.icdf(torch.tensor(conf_high, device=y_pred.device))
        
        in_bin = (z >= lower) & (z < upper)
        if in_bin.any():
            empirical = in_bin.float().mean()
            expected = (conf_high - conf_low)
            ece += (empirical - expected).abs() * in_bin.float().mean() / expected
            
    return ece


def MCE(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    sigma2: torch.Tensor,
    n_bins: int = 15,
) -> torch.Tensor:
    """Maximum Calibration Error."""
    mask = ~torch.isnan(y_true)
    if not mask.any():
        return torch.tensor(float("nan"), device=y_pred.device)
    y_pred = y_pred[mask]
    y_true = y_true[mask]
    sigma2 = sigma2[mask]
    
    sigma = torch.sqrt(sigma2.clamp(min=1e-8))
    z = (y_true - y_pred) / sigma
    
    from torch.distributions import Normal
    normal = Normal(0, 1)
    
    mce = torch.tensor(0.0, device=y_pred.device)
    for i in range(n_bins):
        conf_low = i / n_bins
        conf_high = (i + 1) / n_bins
        lower = normal.icdf(torch.tensor(conf_low, device=y_pred.device))
        upper = normal.icdf(torch.tensor(conf_high, device=y_pred.device))
        
        in_bin = (z >= lower) & (z < upper)
        if in_bin.any():
            empirical = in_bin.float().mean()
            expected = (conf_high - conf_low)
            mce = torch.max(mce, (empirical - expected).abs())
            
    return mce


def TACE(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    sigma2: torch.Tensor,
) -> torch.Tensor:
    """Tail-Adaptive Calibration Error."""
    mask = ~torch.isnan(y_true)
    if not mask.any():
        return torch.tensor(float("nan"), device=y_pred.device)
    y_pred = y_pred[mask]
    y_true = y_true[mask]
    sigma2 = sigma2[mask]
    
    sigma = torch.sqrt(sigma2.clamp(min=1e-8))
    z = (y_true - y_pred) / sigma
    
    from torch.distributions import Normal
    normal = Normal(0, 1)
    
    # Tail bins: [0, 0.1], [0.1, 0.25], [0.25, 0.5], [0.5, 0.75], [0.75, 0.9], [0.9, 1.0]
    bins = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    
    tace = torch.tensor(0.0, device=y_pred.device)
    for i in range(len(bins) - 1):
        conf_low = bins[i]
        conf_high = bins[i + 1]
        lower = normal.icdf(torch.tensor(conf_low, device=y_pred.device))
        upper = normal.icdf(torch.tensor(conf_high, device=y_pred.device))
        
        in_bin = (z >= lower) & (z < upper)
        if in_bin.any():
            empirical = in_bin.float().mean()
            expected = (conf_high - conf_low)
            tace += (empirical - expected).abs() * in_bin.float().mean() / expected
            
    return tace


def Sharpness(sigma2: torch.Tensor) -> torch.Tensor:
    """Average predictive variance (sharpness)."""
    mask = ~torch.isnan(sigma2)
    if not mask.any():
        return torch.tensor(float("nan"), device=sigma2.device)
    return sigma2[mask].mean()


def RoutingAccuracy(
    routed: torch.Tensor,
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    sigma2: torch.Tensor,
) -> torch.Tensor:
    """Accuracy of routing decisions."""
    mask = ~torch.isnan(y_true)
    if not mask.any():
        return torch.tensor(float("nan"), device=routed.device)
    routed = routed[mask]
    return routed.float().mean()


def ThresholdAccuracy(
    routed: torch.Tensor,
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    sigma2: torch.Tensor,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Accuracy of predictions when abstaining based on uncertainty."""
    mask = ~torch.isnan(y_true)
    if not mask.any():
        return torch.tensor(float("nan"), device=routed.device)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    sigma2 = sigma2[mask]
    routed = routed[mask]
    
    # Only evaluate on routed predictions
    if routed.sum() == 0:
        return torch.tensor(0.0, device=y_true.device)
    
    errors = (y_pred - y_true).abs()
    return (errors <= threshold).float().mean()


def RecoveryError(mu_hat: torch.Tensor, mu_true: torch.Tensor) -> torch.Tensor:
    """Mean squared error in recovered statistics."""
    mask = ~torch.isnan(mu_true).any(dim=-1)
    if not mask.any():
        return torch.tensor(float("nan"), device=mu_hat.device)
    mu_hat = mu_hat[mask]
    mu_true = mu_true[mask]
    return F.mse_loss(mu_hat, mu_true)


def EquivarianceError(
    R_composed: torch.Tensor,
    R_target: torch.Tensor,
) -> torch.Tensor:
    """Frobenius norm of intertwiner composition error."""
    return torch.norm(R_composed - R_target, p="fro") / (torch.norm(R_target, p="fro") + 1e-8)


def ManifoldIdempotence(
    rho: callable,
    mu: torch.Tensor,
) -> torch.Tensor:
    """Idempotence error of retraction: ||rho(rho(mu)) - rho(mu)||^2."""
    with torch.no_grad():
        mu_rho = rho(mu)
        mu_rho_rho = rho(mu_rho)
        return F.mse_loss(mu_rho_rho, mu_rho)


def ManifoldFixing(
    rho: callable,
    mu: torch.Tensor,
) -> torch.Tensor:
    """Fixing error: distance from mu(M)."""
    with torch.no_grad():
        mu_rho = rho(mu)
        return F.mse_loss(mu_rho, mu)


class MINE(torch.nn.Module):
    """Mutual Information Neural Estimation (MINE)."""
    
    def __init__(self, x_dim: int, y_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(x_dim + y_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (MI estimate, T(x,y))."""
        batch_size = x.size(0)
        
        # Joint samples
        joint = torch.cat([x, y], dim=1)
        T_joint = self.network(joint)
        
        # Marginal samples (shuffle y)
        y_perm = y[torch.randperm(batch_size)]
        marginal = torch.cat([x, y_perm], dim=1)
        T_marginal = self.network(marginal)
        
        # Donsker-Varadhan lower bound
        mi_estimate = T_joint.mean() - torch.log(torch.exp(T_marginal).mean() + 1e-8)
        
        return mi_estimate, T_joint


class UncertaintyDecomposition:
    """Decompose total uncertainty into aleatoric, epistemic, non-identifiability."""
    
    @staticmethod
    def decompose(
        sigma2_aleat: torch.Tensor,
        sigma2_epi: torch.Tensor,
        sigma2_nonid: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        sigma2_total = sigma2_aleat + sigma2_epi + sigma2_nonid
        return {
            "sigma2_aleat": sigma2_aleat,
            "sigma2_epi": sigma2_epi,
            "sigma2_nonid": sigma2_nonid,
            "sigma2_total": sigma2_total,
            "frac_aleat": sigma2_aleat / (sigma2_total + 1e-8),
            "frac_epi": sigma2_epi / (sigma2_total + 1e-8),
            "frac_nonid": sigma2_nonid / (sigma2_total + 1e-8),
        }
    
    @staticmethod
    def from_ensemble(
        predictions: torch.Tensor,  # [ensemble_size, batch]
        sigma2_aleat: torch.Tensor,  # [batch]
    ) -> Dict[str, torch.Tensor]:
        """Estimate uncertainty decomposition from ensemble predictions."""
        # Epistemic: variance across ensemble
        sigma2_epi = predictions.var(dim=0, unbiased=False)
        
        # Aleatoric: average of individual variances
        sigma2_aleat = sigma2_aleat
        
        # Total
        sigma2_total = predictions.mean(dim=0).var(dim=0, unbiased=False) + sigma2_aleat.mean()
        
        # Non-identifiability = total - aleat - epi
        sigma2_nonid = (sigma2_total - sigma2_aleat - sigma2_epi).clamp(min=0)
        
        return UncertaintyDecomposition.decompose(
            sigma2_aleat, sigma2_epi, sigma2_nonid
        )


def compute_all_metrics(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    sigma2: Optional[torch.Tensor] = None,
    sigma2_aleat: Optional[torch.Tensor] = None,
    sigma2_epi: Optional[torch.Tensor] = None,
    sigma2_nonid: Optional[torch.Tensor] = None,
    routed: Optional[torch.Tensor] = None,
    mu_hat: Optional[torch.Tensor] = None,
    mu_true: Optional[torch.Tensor] = None,
    R_composed: Optional[torch.Tensor] = None,
    R_target: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """Compute all available metrics."""
    metrics = {}
    
    # Regression
    metrics["nmse"] = NMSE(y_pred, y_true)
    
    # Calibration (if sigma2 provided)
    if sigma2 is not None:
        metrics["nll"] = NLL(y_pred, y_true, sigma2)
        metrics["ece"] = ECE(y_pred, y_true, sigma2)
        metrics["mce"] = MCE(y_pred, y_true, sigma2)
        metrics["tace"] = TACE(y_pred, y_true, sigma2)
        metrics["sharpness"] = Sharpness(sigma2)
    
    # Gate accuracy (if routed provided)
    if routed is not None:
        metrics["routing_accuracy"] = RoutingAccuracy(routed, y_true, y_pred, sigma2 if sigma2 is not None else torch.ones_like(y_true))
        metrics["threshold_accuracy_1"] = ThresholdAccuracy(routed, y_true, y_pred, sigma2 if sigma2 is not None else torch.ones_like(y_true), threshold=1.0)
        metrics["threshold_accuracy_2"] = ThresholdAccuracy(routed, y_true, y_pred, sigma2 if sigma2 is not None else torch.ones_like(y_true), threshold=2.0)
    
    # Recovery error (if mu_true provided)
    if mu_hat is not None and mu_true is not None:
        metrics["recovery_error"] = RecoveryError(mu_hat, mu_true)
    
    # Equivariance error (if R matrices provided)
    if R_composed is not None and R_target is not None:
        metrics["equivariance_error"] = EquivarianceError(R_composed, R_target)
    
    # Uncertainty decomposition
    if sigma2_aleat is not None and sigma2_epi is not None and sigma2_nonid is not None:
        decomp = UncertaintyDecomposition.decompose(sigma2_aleat, sigma2_epi, sigma2_nonid)
        metrics.update(decomp)
    
    return metrics
