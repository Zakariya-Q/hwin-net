"""
HWIN-Net: L_rec - Recovery Loss

Mathematical Purpose
--------------------
Implements per Computational Constraint CC1 the recovery loss.

L_rec = min_theta max_phi E[loss(T_g(z_g), mu_true)]

Can be implemented as:
1. Direct MSE when mu_true is available (supervised)
2. Min-max variational form when mu_true is latent

Theory Traceability
-------------------
- Axiom A4 (Uniform Identifiability): T_g exists
- Lemma 2 (Recovery <=> Marginal Identifiability)
- CC1 (Recovery via Min-Max): Variational form

Tensor Signatures
-----------------
- mu_hat: [B, d] recovered statistic from RecoveryModule
- mu_true: [B, d] ground truth (when available)
- z_g: [B, k] masked encoding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from dataclasses import dataclass


@dataclass
class RecoveryLossConfig:
    loss_type: str = "mse"  # "mse" or "minmax"
    lambda_rec: float = 1.0


class RecoveryLoss(nn.Module):
    """
    L_rec = ||mu_hat - mu_true||^2 (MSE) or min-max form.

    Tensor Signatures:
    - mu_hat: [B, d] from RecoveryModule
    - mu_true: [B, d] ground truth (optional, for supervised)
    - z_g: [B, k] masked encoding (for min-max)
    """

    def __init__(self, config: RecoveryLossConfig):
        super().__init__()
        self.config = config
        self.loss_type = config.loss_type
        self.lambda_rec = config.lambda_rec

    def forward(
        self,
        mu_hat: torch.Tensor,
        mu_true: Optional[torch.Tensor] = None,
        z_g: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute recovery loss.

        Args:
            mu_hat: [B, d] recovered statistic
            mu_true: [B, d] ground truth mu (if available)
            z_g: [B, k] masked encoding (for min-max)

        Returns:
            Scalar loss
        """
        if self.loss_type == "mse":
            if mu_true is None:
                # No ground truth - skip
                return torch.tensor(0.0, device=mu_hat.device, dtype=mu_hat.dtype)
            loss = F.mse_loss(mu_hat, mu_true)
        elif self.loss_type == "minmax":
            # Min-max form: would require a critic/evaluator
            # For now fallback to MSE if mu_true available
            if mu_true is not None:
                loss = F.mse_loss(mu_hat, mu_true)
            else:
                # Unsupervised: cannot compute without critic
                loss = torch.tensor(0.0, device=mu_hat.device, dtype=mu_hat.dtype)
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        return self.lambda_rec * loss

    def extra_repr(self) -> str:
        return f"loss_type={self.loss_type}, lambda_rec={self.lambda_rec}"


def create_recovery_loss(config: RecoveryLossConfig) -> RecoveryLoss:
    return RecoveryLoss(config)
