"""
HWIN-Net: L_noleak - No-Leakage MI Loss

Mathematical Purpose
--------------------
Implements the no-leakage regularizer per Axiom A5 (No Leakage),
Computational Constraint CC5 (MI Penalty), and Conjecture C1 (Converse).

L_noleak = lambda_MI * I(z_g; a | O)  (estimated via adversarial or MINE)

Theory Traceability
-------------------
- Axiom A5 (No Leakage): I(z_g; a | O) = 0
- CC5: L_noleak = lambda_MI * I(z_g; a)
- C1 (Converse): If I(z_g; a) = 0, no platform info leakage
- Lemma L8 (Factorization): psi(mu_F) independent of schema

Tensor Signatures
-----------------
- z_g: [B, k] masked encoding
- a_idx: [B] platform indices
- Returns: scalar MI loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class NoLeakageLossConfig:
    mi_estimator: str = "adversarial"  # "adversarial" or "mine"
    lambda_mi: float = 0.1


class NoLeakageLoss(nn.Module):
    """
    L_noleak = lambda_MI * I(z_g; a)

    Uses the NoLeakageRegularizer module which implements both
    adversarial (DANN/GRL) and MINE estimators.

    Tensor Signatures:
    - z_g: [B, k] masked encoding from SchemaEncoder
    - a_idx: [B] platform indices
    - Returns: dict with 'noleak_loss', 'disc_loss', 'disc_acc', 'mi_estimate'
    """

    def __init__(self, config: NoLeakageLossConfig, z_dim: int, num_platforms: int):
        super().__init__()
        self.config = config
        self.lambda_mi = config.lambda_mi

        # Import here to avoid circular imports
        from hwin_net.models.no_leakage import NoLeakageRegularizer, NoLeakageRegularizerConfig

        no_leakage_config = NoLeakageRegularizerConfig(
            mi_estimator=config.mi_estimator,
            lambda_mi=config.lambda_mi,
        )
        self.regularizer = NoLeakageRegularizer(
            no_leakage_config,
            z_dim=z_dim,
            num_platforms=num_platforms,
        )

    def forward(
        self,
        z_g: torch.Tensor,
        a_idx: torch.Tensor,
        training: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Compute no-leakage loss.

        Args:
            z_g: [B, k] masked encoding
            a_idx: [B] platform indices
            training: bool, whether to compute loss (True) or return zeros (False)

        Returns:
            Dict with loss and auxiliary metrics
        """
        out = self.regularizer(z_g, a_idx, training=training)
        # The regularizer already applies lambda_mi
        return out

    def discriminator_step(self, z_g: torch.Tensor, a_idx: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Update discriminator only."""
        return self.regularizer.discriminator_step(z_g, a_idx)

    def encoder_step(self, z_g: torch.Tensor, a_idx: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Update encoder to minimize MI."""
        return self.regularizer.encoder_step(z_g, a_idx)

    def extra_repr(self) -> str:
        return f"mi_estimator={self.config.mi_estimator}, lambda_mi={self.lambda_mi}"


def create_noleak_loss(
    config: NoLeakageLossConfig,
    z_dim: int,
    num_platforms: int
) -> NoLeakageLoss:
    return NoLeakageLoss(config, z_dim, num_platforms)
