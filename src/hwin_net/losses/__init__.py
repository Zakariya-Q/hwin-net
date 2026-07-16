"""
HWIN-Net: Losses Package
=========================

Loss functions implementing the theoretical objectives:
- L_pred: Prediction loss (MSE/NLL with uncertainty)
- L_rec: Recovery loss (supervised/min-max)
- L_no-leak: No-leakage loss (adversarial MI estimation)
- L_equiv: Equivariance loss (intertwiner consistency)
- L_complex: Complexity regularization
- TotalLoss: Combined loss with configurable weights
"""

from hwin_net.losses.prediction_loss import PredictionLoss, PredictionLossConfig, create_prediction_loss
from hwin_net.losses.recovery_loss import RecoveryLoss, RecoveryLossConfig, create_recovery_loss
from hwin_net.losses.noleak_loss import NoLeakageLoss, NoLeakageLossConfig, create_noleak_loss
from hwin_net.losses.equivariance_loss import EquivarianceLoss, EquivarianceLossConfig, create_equivariance_loss
from hwin_net.losses.complexity_loss import ComplexityLoss, ComplexityLossConfig, create_complexity_loss
from hwin_net.losses.total_loss import TotalLoss, TotalLossConfig, create_total_loss

__all__ = [
    "PredictionLoss",
    "PredictionLossConfig",
    "create_prediction_loss",
    "RecoveryLoss",
    "RecoveryLossConfig",
    "create_recovery_loss",
    "NoLeakageLoss",
    "NoLeakageLossConfig",
    "create_noleak_loss",
    "EquivarianceLoss",
    "EquivarianceLossConfig",
    "create_equivariance_loss",
    "ComplexityLoss",
    "ComplexityLossConfig",
    "create_complexity_loss",
    "TotalLoss",
    "TotalLossConfig",
    "create_total_loss",
]
