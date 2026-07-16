import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List
from dataclasses import dataclass


@dataclass
class TotalLossConfig:
    lambda_pred: float = 1.0
    lambda_rec: float = 1.0
    lambda_noleak: float = 0.1
    lambda_equiv: float = 0.1
    lambda_complex: float = 1e-4
    # Loss types
    pred_loss_type: str = "mse"
    rec_loss_type: str = "mse"
    equiv_loss_type: str = "frobenius"


class TotalLoss(nn.Module):
    def __init__(
        self,
        config: TotalLossConfig,
        z_dim: int = 128,
        num_platforms: int = 3,
        latent_dim: int = 64,
    ):
        super().__init__()
        self.config = config
        from hwin_net.losses.prediction_loss import PredictionLoss, PredictionLossConfig
        from hwin_net.losses.recovery_loss import RecoveryLoss, RecoveryLossConfig
        from hwin_net.losses.noleak_loss import NoLeakageLoss, NoLeakageLossConfig
        from hwin_net.losses.equivariance_loss import EquivarianceLoss, EquivarianceLossConfig
        from hwin_net.losses.complexity_loss import ComplexityLoss, ComplexityLossConfig

        self.pred_loss = PredictionLoss(PredictionLossConfig(
            loss_type=config.pred_loss_type,
            lambda_pred=config.lambda_pred,
        ))

        self.rec_loss = RecoveryLoss(RecoveryLossConfig(
            loss_type=config.rec_loss_type,
            lambda_rec=config.lambda_rec,
        ))

        self.noleak_loss = NoLeakageLoss(NoLeakageLossConfig(
            mi_estimator="adversarial",
            lambda_mi=config.lambda_noleak,
        ), z_dim=z_dim, num_platforms=num_platforms)

        self.equiv_loss = EquivarianceLoss(EquivarianceLossConfig(
            equiv_loss_type=config.equiv_loss_type,
            lambda_equiv=config.lambda_equiv,
        ))

        self.complex_loss = ComplexityLoss(ComplexityLossConfig(
            norm_type="l2",
            lambda_complex=config.lambda_complex,
        ))

        self.active_losses = {
            "pred": config.lambda_pred > 0,
            "rec": config.lambda_rec > 0,
            "noleak": config.lambda_noleak > 0,
            "equiv": config.lambda_equiv > 0,
            "complex": config.lambda_complex > 0,
        }

    def forward(
        self,
        q_out: torch.Tensor,
        q_hat: torch.Tensor,
        sigma2_aleat: Optional[torch.Tensor],
        sigma2_total: Optional[torch.Tensor],
        y: Optional[torch.Tensor],
        mu_hat: Optional[torch.Tensor] = None,
        mu_final: Optional[torch.Tensor] = None,
        mu_true: Optional[torch.Tensor] = None,
        z_g: Optional[torch.Tensor] = None,
        a_idx: Optional[torch.Tensor] = None,
        model: Optional[torch.nn.Module] = None,
        training: bool = True,
    ) -> Dict[str, torch.Tensor]:
        losses = {}

        if self.active_losses["pred"] and y is not None:
            pred_loss = self.pred_loss(q_out, y, sigma2_total)
            losses["pred_loss"] = pred_loss
        else:
            losses["pred_loss"] = torch.tensor(0.0, device=q_out.device, dtype=q_out.dtype)

        if self.active_losses["rec"] and mu_true is not None and mu_hat is not None:
            rec_loss = self.rec_loss(mu_hat, mu_true, z_g)
            losses["rec_loss"] = rec_loss
        else:
            losses["rec_loss"] = torch.tensor(0.0, device=q_out.device, dtype=q_out.dtype)

        if self.active_losses["noleak"] and training and z_g is not None and a_idx is not None:
            noleak_out = self.noleak_loss(z_g, a_idx, training=True)
            losses["noleak_loss"] = noleak_out["noleak_loss"]
            for k, v in noleak_out.items():
                if k != "noleak_loss" and isinstance(v, torch.Tensor):
                    losses[k] = v
        else:
            losses["noleak_loss"] = torch.tensor(0.0, device=q_out.device, dtype=q_out.dtype)

        if self.active_losses["equiv"] and training and z_g is not None and a_idx is not None:
            recovery_module = model.recovery_module if model is not None else None
            equiv_loss = self.equiv_loss(z_g, a_idx, recovery_module)
            losses["equiv_loss"] = equiv_loss
        else:
            losses["equiv_loss"] = torch.tensor(0.0, device=q_out.device, dtype=q_out.dtype)

        if self.active_losses["complex"] and mu_final is not None:
            complex_loss = self.complex_loss(mu_final)
            losses["complex_loss"] = complex_loss
        else:
            losses["complex_loss"] = torch.tensor(0.0, device=q_out.device, dtype=q_out.dtype)

        total_loss = sum(
            losses[k] for k in losses
            if "loss" in k and k != "total_loss"
        )
        losses["total_loss"] = total_loss

        return losses

    def forward_inference(
        self,
        q_out: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        return self.forward(
            q_out=q_out,
            q_hat=q_out,
            sigma2_aleat=None,
            sigma2_total=None,
            y=y,
            training=False,
        )

    def extra_repr(self) -> str:
        active = [k for k, v in self.active_losses.items() if v]
        return f"active_losses={active}"


def create_total_loss(
    config: TotalLossConfig,
    z_dim: int = 128,
    num_platforms: int = 3,
    latent_dim: int = 64,
) -> TotalLoss:
    return TotalLoss(config, z_dim, num_platforms, latent_dim)
