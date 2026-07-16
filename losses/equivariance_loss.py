"""
HWIN-Net: L_equiv - Equivariance Loss

Mathematical Purpose
--------------------
Implements the equivariance constraint from Lemma 5 and CC3.

Per Lemma 5: For fixed O, T_{(O,a2)} = R_{a2,a1} o T_{(O,a1)}
Per CC3: Weight sharing via intertwiner R_{a2,a1}

Theory Traceability
-------------------
- Lemma 5 (Equivariance Structure): Intertwiners compose groupoidally
- CC3 (Equivariance Constraint): Weight sharing via R_{a2,a1}
- R_{a,a} = Identity
- R_{a3,a1} = R_{a3,a2} o R_{a2,a1} (groupoid composition)

Tensor Signatures
-----------------
- z_g: [B, k] masked encoding
- a_idx: [B] platform indices
- Returns: scalar loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List
from dataclasses import dataclass

from models.recovery_module import RecoveryModule, RecoveryConfig


@dataclass
class EquivarianceLossConfig:
    loss_type: str = "frobenius"  # "frobenius", "spectral", "mse"
    lambda_equiv: float = 0.1

    def __init__(self, loss_type: str = "frobenius", lambda_equiv: float = 0.1, equiv_loss_type: str = None):
        self.loss_type = loss_type
        self.lambda_equiv = lambda_equiv
        # Allow equiv_loss_type as alias for loss_type (for TotalLossConfig compatibility)
        if equiv_loss_type is not None:
            self.loss_type = equiv_loss_type


class EquivarianceLoss(nn.Module):
    """
    L_equiv = ||R_{a2,a1} - I||_F^2 (frobenius) or spectral norm

    For shared T_base + intertwiners: enforces R_{a2,a1} approx I on data subspace
    For separate T_g: enforces T_{(O,a2)}(z) = R_{a2,a1}(T_{(O,a1)}(z))

    Tensor Signatures:
    - z_g: [B, k] masked encoding
    - a_idx: [B] platform indices
    """

    def __init__(self, config: EquivarianceLossConfig):
        super().__init__()
        self.config = config
        self.loss_type = config.loss_type
        self.lambda_equiv = config.lambda_equiv

    def forward(
        self,
        z_g: torch.Tensor,
        a_idx: torch.Tensor,
        recovery_module: RecoveryModule
    ) -> torch.Tensor:
        """
        Compute equivariance loss using RecoveryModule's built-in method.

        Args:
            z_g: [B, k] masked encoding
            a_idx: [B] platform indices
            recovery_module: The RecoveryModule instance with intertwiners

        Returns:
            Weighted scalar loss
        """
        if recovery_module is None:
            # Fallback: return 0 loss if recovery_module not provided
            return torch.tensor(0.0, device=z_g.device, dtype=z_g.dtype)
        if self.loss_type == "frobenius":
            # Use RecoveryModule's equivariance_loss which computes ||R_{a2,a1} - I||_F^2
            loss = recovery_module.equivariance_loss(z_g, a_idx)
        elif self.loss_type == "mse":
            # Simpler: enforce all platforms produce same latent after base + intertwiner
            loss = recovery_module.equivariance_loss_simple(z_g, a_idx)
        elif self.loss_type == "spectral":
            loss = self._spectral_equivariance_loss(z_g, a_idx, recovery_module)
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        return self.lambda_equiv * loss

    def _spectral_equivariance_loss(
        self,
        z_g: torch.Tensor,
        a_idx: torch.Tensor,
        recovery_module: RecoveryModule
    ) -> torch.Tensor:
        """Spectral norm of difference from identity for intertwiners."""
        loss = torch.tensor(0.0, device=z_g.device, dtype=z_g.dtype)
        count = 0

        platforms_present = a_idx.unique().tolist()
        if len(platforms_present) < 2:
            return loss

        base = recovery_module.config.base_platform
        for a in platforms_present:
            if a == base:
                continue
            R_a = recovery_module.get_intertwiner(a)
            if hasattr(R_a, 'transform') and hasattr(R_a.transform, 'weight'):
                W = R_a.transform.weight
                # Spectral norm of W - I
                W_centered = W - torch.eye(W.shape[0], device=W.device, dtype=W.dtype)
                # Approximate spectral norm via power iteration
                eigvals = torch.linalg.eigvals(W_centered)
                spectral_norm = eigvals.abs().max()
                loss += spectral_norm
                count += 1

        if count > 0:
            loss = loss / count
        return loss

    def extra_repr(self) -> str:
        return f"loss_type={self.loss_type}, lambda_equiv={self.lambda_equiv}"


def create_equivariance_loss(config: EquivarianceLossConfig) -> EquivarianceLoss:
    return EquivarianceLoss(config)
