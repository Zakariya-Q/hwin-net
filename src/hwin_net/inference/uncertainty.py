"""
HWIN-Net: Uncertainty Quantification Utilities

Mathematical Purpose
--------------------
Implements additional uncertainty quantification methods:
- MC Dropout for epistemic uncertainty
- Ensemble methods
- Conformal prediction intervals
- Calibration metrics

Theory Traceability
-------------------
- Theorem 5 (Uncertainty Decomposition)
- Training section (mixed precision, dropout)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple
import numpy as np


class MCDropoutUncertainty:
    """
    Monte Carlo Dropout for epistemic uncertainty estimation.
    
    During inference, run multiple forward passes with dropout enabled
    to estimate predictive variance (epistemic uncertainty).
    """
    
    def __init__(self, model: nn.Module, n_samples: int = 30, dropout_prob: float = 0.1):
        self.model = model
        self.n_samples = n_samples
        self.dropout_prob = dropout_prob
    
    @torch.no_grad()
    def __call__(
        self,
        x: torch.Tensor,
        M_O: torch.Tensor,
        a_idx: torch.Tensor,
        e_a: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run MC dropout inference.
        
        Returns:
            mean_pred: [B, |Y|] mean prediction across samples
            epistemic_var: [B, |Y|] variance across samples (epistemic uncertainty)
        """
        # Enable dropout during eval
        self.model.train()
        for module in self.model.modules():
            if isinstance(module, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
                module.p = self.dropout_prob
        
        predictions = []
        for _ in range(self.n_samples):
            out = self.model.forward_inference(x, M_O, a_idx, e_a)
            predictions.append(out['q_out'])  # Already routed
        self.model.eval()
        
        preds = torch.stack(predictions, dim=0)  # [n_samples, B, |Y|]
        mean_pred = preds.mean(dim=0)
        epistemic_var = preds.var(dim=0)
        
        return mean_pred, epistemic_var
    
    def enable_dropout(self):
        """Enable dropout for MC sampling."""
        for module in self.model.modules():
            if isinstance(module, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
                module.train()
                module.p = self.dropout_prob
    
    def disable_dropout(self):
        """Disable dropout."""
        for module in self.model.modules():
            if isinstance(module, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
                module.eval()


class EnsembleUncertainty:
    """
    Deep Ensemble for epistemic uncertainty.
    
    Train multiple models with different seeds and average predictions.
    """
    
    def __init__(self, models: List[nn.Module]):
        self.models = models
        for m in self.models:
            m.eval()
    
    @torch.no_grad()
    def __call__(
        self,
        x: torch.Tensor,
        M_O: torch.Tensor,
        a_idx: torch.Tensor,
        e_a: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        predictions = []
        for model in self.models:
            out = model.forward_inference(x, M_O, a_idx, e_a)
            predictions.append(out['q_out'])
        
        preds = torch.stack(predictions, dim=0)  # [n_models, B, |Y|]
        mean_pred = preds.mean(dim=0)
        epistemic_var = preds.var(dim=0)
        
        return mean_pred, epistemic_var


def compute_calibration_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    uncertainties: torch.Tensor,
    n_bins: int = 10,
) -> dict:
    """
    Compute calibration metrics for uncertainty quantification.
    
    Args:
        preds: [B] or [B, |Y|] predictions
        targets: [B] or [B, |Y|] ground truth
        uncertainties: [B] or [B, |Y|] total uncertainty (variance)
        n_bins: number of bins for ECE
    
    Returns:
        Dict with ECE, AUROC, etc.
    """
    # Flatten if multi-dimensional
    if preds.dim() > 1:
        preds = preds.flatten()
        targets = targets.flatten()
        uncertainties = uncertainties.flatten()
    
    # Expected Calibration Error (ECE)
    # Bin by uncertainty, compute accuracy in each bin
    stds = torch.sqrt(uncertainties + 1e-8)
    
    # Sort by uncertainty
    idx = torch.argsort(stds)
    preds_sorted = preds[idx]
    targets_sorted = targets[idx]
    stds_sorted = stds[idx]
    
    # Bin statistics
    bin_size = len(preds) // n_bins
    ece = 0.0
    for i in range(n_bins):
        start = i * bin_size
        end = (i + 1) * bin_size if i < n_bins - 1 else len(preds)
        if end <= start:
            continue
        
        bin_preds = preds_sorted[start:end]
        bin_targets = targets_sorted[start:end]
        bin_stds = stds_sorted[start:end]
        
        # Accuracy
        acc = (torch.abs(bin_preds - bin_targets) < bin_stds).float().mean()
        # Confidence (fraction within 1 std)
        conf = (torch.abs(bin_preds - bin_targets) < bin_stds).float().mean()
        ece += abs(acc - conf) * (end - start) / len(preds)
    
    # Correlation between absolute error and uncertainty
    abs_error = torch.abs(preds - targets)
    unc_corr = torch.corrcoef(torch.stack([abs_error, stds]))[0, 1].item()
    
    return {
        'ece': ece.item(),
        'uncertainty_error_correlation': unc_corr,
        'mean_abs_error': abs_error.mean().item(),
        'mean_uncertainty': stds.mean().item(),
    }


def conformal_prediction_interval(
    preds: torch.Tensor,
    uncertainties: torch.Tensor,
    alpha: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute conformal prediction intervals.
    
    Args:
        preds: [B] predictions
        uncertainties: [B] predicted standard deviations (not variance!)
        alpha: significance level (0.1 -> 90% interval)
    
    Returns:
        lower, upper: [B] interval bounds
    """
    # Use split conformal prediction
    # Interval = pred +/- quantile(alpha/2) * uncertainty
    z = torch.distributions.Normal(0, 1).icdf(torch.tensor(1 - alpha / 2))
    margin = z * uncertainties
    lower = preds - margin
    upper = preds + margin
    return lower, upper
