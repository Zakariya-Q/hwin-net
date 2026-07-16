"""
HWIN-Net: Prediction Loss (L_pred)

Theory Traceability
-------------------
- Axiom A4 (Uniform Identifiability) -> mu exists
- Theorem 3 (SIS Existence) -> q = psi(mu_F)
- Theorem 5 (SIS Uncertainty Decomposition) -> sigma^2_aleat, sigma^2_epi
- Lemma 8 (Factorization) -> q = psi(mu_F) depends only on mu_F, not schema

Tensor Signatures
-----------------
- q_hat: [B, |Y|] or [B] predictions
- y: [B, |Y|] or [B] targets
- sigma2_aleat: [B, |Y|] or [B] aleatoric variance
- sigma2_total: [B, |Y|] or [B] total variance (ale + epi + nonid)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from dataclasses import dataclass


@dataclass
class PredictionLossConfig:
    loss_type: str = "mse"  # "mse", "nll", "gaussian_nll"
    lambda_pred: float = 1.0


class PredictionLoss(nn.Module):
    """
    L_pred = ||q_hat - y||^2 (MSE) or NLL with uncertainty

    For Gaussian NLL:
        L_pred = 0.5 * (log(sigma2_total) + (q_hat - y)^2 / sigma2_total)

    Tensor Signatures:
    - q_hat: [B] or [B, |Y|]
    - y: [B] or [B, |Y|]
    - sigma2_total: [B] or [B, |Y|] (aleatoric + epistemic + non-identifiability)
    - sigma2_aleat: [B] or [B, |Y|] (aleatoric only)
    """

    def __init__(self, config: PredictionLossConfig):
        super().__init__()
        self.config = config
        self.loss_type = config.loss_type
        self.lambda_pred = config.lambda_pred

    def forward(
        self,
        q_hat: torch.Tensor,
        y: torch.Tensor,
        sigma2_total: Optional[torch.Tensor] = None,
        sigma2_aleat: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute prediction loss.

        Args:
            q_hat: [B] or [B, |Y|] predictions
            y: [B] or [B, |Y|] targets
            sigma2_total: [B] or [B, |Y|] total variance (optional, for NLL)
            sigma2_aleat: [B] or [B, |Y|] aleatoric variance (optional, for NLL)
        """
        # Mask out NaN targets
        valid_mask = ~torch.isnan(y)
        if not valid_mask.any():
            return torch.tensor(0.0, device=q_hat.device, dtype=q_hat.dtype)

        q_hat_valid = q_hat[valid_mask]
        y_valid = y[valid_mask]
        
        if sigma2_total is not None:
            sigma2_total_valid = sigma2_total[valid_mask]
        else:
            sigma2_total_valid = None
        if sigma2_aleat is not None:
            sigma2_aleat_valid = sigma2_aleat[valid_mask]
        else:
            sigma2_aleat_valid = None

        if self.loss_type == "mse":
            loss = F.mse_loss(q_hat_valid, y_valid)

        elif self.loss_type in ("nll", "gaussian_nll"):
            if sigma2_total_valid is None:
                loss = F.mse_loss(q_hat_valid, y_valid)
            else:
                eps = 1e-6
                sigma2 = sigma2_total_valid.clamp(min=eps)
                loss = 0.5 * (torch.log(sigma2) + (q_hat_valid - y_valid) ** 2 / sigma2).mean()

        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        return self.lambda_pred * loss

    def extra_repr(self) -> str:
        return f"loss_type={self.loss_type}, lambda_pred={self.lambda_pred}"


def create_prediction_loss(config: PredictionLossConfig) -> PredictionLoss:
    return PredictionLoss(config)
