"""
HWIN-Net: L_complex - Complexity Penalty on mu

Mathematical Purpose
--------------------
Implements per CC1 (Min-Max Optimization) the L2 penalty on mu_final
to prevent overfitting and control model capacity.

Theory Traceability
-------------------
- CC1 (Recovery via Min-Max): Complexity penalty on mu
- Axiom A4 (Uniform Identifiability): Controls mu dimension

Tensor Signatures
-----------------
- mu_final: [B, d] retracted statistic on manifold
- Returns: scalar loss
"""

import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class ComplexityLossConfig:
    norm_type: str = "l2"  # "l2", "l1" (not currently used)
    lambda_complex: float = 1e-4


class ComplexityLoss(nn.Module):
    """
    L_complex = lambda_complex * ||mu_final||^2

    Penalizes large latent codes to control capacity.

    Tensor Signatures:
    - mu_final: [B, d]
    """

    def __init__(self, config: ComplexityLossConfig):
        super().__init__()
        self.config = config
        self.lambda_complex = config.lambda_complex

    def forward(self, mu_final: torch.Tensor) -> torch.Tensor:
        """
        Compute complexity loss.

        Args:
            mu_final: [B, d] retracted statistic
        """
        if self.config.norm_type == "l2":
            loss = mu_final.pow(2).mean()
        elif self.config.norm_type == "l1":
            loss = mu_final.abs().mean()
        else:
            raise ValueError(f"Unknown norm_type: {self.config.norm_type}")

        return self.lambda_complex * loss

    def extra_repr(self) -> str:
        return f"norm_type={self.config.norm_type}, lambda_complex={self.lambda_complex}"


def create_complexity_loss(config: ComplexityLossConfig) -> ComplexityLoss:
    return ComplexityLoss(config)
