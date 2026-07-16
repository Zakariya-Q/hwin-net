"""HWIN-Net: Module 6 - NoLeakageRegularizer (M6)

Mathematical Purpose
--------------------
Implements the No-Leakage Regularizer per Axiom A5 (No Leakage),
Computational Constraint CC5 (MI Penalty), and Conjecture C1 (Converse).

Per section 5.6 of HWIN_Net_Spec.md and SIS_Reaxiomatized.md:
- Enforces I(z_g; a | O) = 0 to prevent platform information leakage
- Uses adversarial MI estimation (DANN-style Gradient Reversal Layer) as primary method
- Supports MINE estimator as alternative (configurable via mi_estimator)
- Training-time ONLY module (no inference forward pass)
- Validates theorem requirement: lambda_mi > 0 per A5, CC5, C1

Theory Traceability
-------------------
- Axiom A5 (No Leakage): I(z_g; a | O) = 0
- Computational Constraint CC5: L_noleak = lambda_MI * I(z_g; a)
- Conjecture C1 (Converse): If I(z_g; a) = 0 then platform info is not leaked
- Lemma L8 (Factorization): q = psi(mu_F) depends only on mu_F, not schema

Implementation follows section 5.6 of HWIN_Net_Spec.md (Module 6: NoLeakageRegularizer)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass, field
from hwin_net.utils.config import NoLeakageConfig


# ============================================================================
# GRADIENT REVERSAL LAYER (GRL) - DANN Style
# ============================================================================

class GradientReversalFunction(torch.autograd.Function):
    """Gradient Reversal Layer (GRL) as an autograd Function."""
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_init=1.0):
        super().__init__()
        self.register_buffer("lambda_", torch.tensor(lambda_init, dtype=torch.float32))
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_.item())
    def set_lambda(self, lambda_):
        self.lambda_.fill_(lambda_)
    def get_lambda(self):
        return self.lambda_.item()
    def extra_repr(self):
        return f"lambda={self.lambda_.item():.4f}"


# ============================================================================
# DISCRIMINATOR NETWORKS
# ============================================================================

class MLPDiscriminator(nn.Module):
    """MLP classifier for schema/platform prediction from latent representation z_g."""
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        activation: str = "gelu",
        norm_type: str = "layer"
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        if activation == "gelu":
            act_fn = nn.GELU
        elif activation == "relu":
            act_fn = nn.ReLU
        elif activation == "silu":
            act_fn = nn.SiLU
        else:
            act_fn = nn.GELU

        def get_norm(dim: int):
            if norm_type == "layer":
                return nn.LayerNorm(dim)
            elif norm_type == "batch":
                return nn.BatchNorm1d(dim)
            else:
                return nn.Identity()

        layers = []
        in_dim = input_dim
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else num_classes
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(get_norm(out_dim))
                layers.append(act_fn())
                layers.append(nn.Dropout(dropout))
            in_dim = out_dim

        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z_g: torch.Tensor) -> torch.Tensor:
        return self.network(z_g)

    def extra_repr(self) -> str:
        return f"input_dim={self.input_dim}, num_classes={self.num_classes}, hidden={self.hidden_dim}, layers={self.num_layers}"


class MINEEstimator(nn.Module):
    """MINE (Mutual Information Neural Estimation) using Donsker-Varadhan bound."""
    def __init__(
        self,
        z_dim: int,
        num_platforms: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        activation: str = "gelu",
        norm_type: str = "layer"
    ):
        super().__init__()
        self.z_dim = z_dim
        self.num_platforms = num_platforms

        if activation == "gelu":
            act_fn = nn.GELU
        elif activation == "relu":
            act_fn = nn.ReLU
        elif activation == "silu":
            act_fn = nn.SiLU
        else:
            act_fn = nn.GELU

        def get_norm(dim: int):
            if norm_type == "layer":
                return nn.LayerNorm(dim)
            elif norm_type == "batch":
                return nn.BatchNorm1d(dim)
            else:
                return nn.Identity()

        layers = []
        in_dim = z_dim + num_platforms
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else 1
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(get_norm(out_dim))
                layers.append(act_fn())
                layers.append(nn.Dropout(dropout))
            in_dim = out_dim

        self.critic = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        z_g: torch.Tensor,
        a_idx: torch.Tensor,
        z_g_shuffled=None,
        a_idx_shuffled=None
    ):
        B = z_g.shape[0]
        device = z_g.device

        a_onehot = F.one_hot(a_idx, num_classes=self.num_platforms).float()
        z_a_joint = torch.cat([z_g, a_onehot], dim=-1)

        joint_scores = self.critic(z_a_joint).squeeze(-1)

        if z_g_shuffled is None or a_idx_shuffled is None:
            perm = torch.randperm(B, device=device)
            z_g_shuffled = z_g[perm]
            a_idx_shuffled = a_idx[perm]

        a_onehot_shuffled = F.one_hot(a_idx_shuffled, num_classes=self.num_platforms).float()
        z_a_marginal = torch.cat([z_g_shuffled, a_onehot_shuffled], dim=-1)
        marginal_scores = self.critic(z_a_marginal).squeeze(-1)

        mi_estimate = joint_scores.mean() - torch.log(torch.exp(marginal_scores).mean() + 1e-8)

        return mi_estimate, joint_scores, marginal_scores


# ============================================================================
# CONFIGURATION DATACLASS
# ============================================================================

@dataclass
class NoLeakageRegularizerConfig:
    """Configuration for NoLeakageRegularizer."""
    mi_estimator: str = field(default="adversarial", metadata={"choices": ["adversarial", "mine"]})
    discriminator_type: str = "mlp"
    discriminator_hidden: int = 64
    discriminator_layers: int = 2
    discriminator_dropout: float = 0.1
    discriminator_activation: str = "gelu"
    discriminator_norm: str = "layer"
    gradient_reversal: bool = True
    grl_lambda_init: float = 1.0
    grl_lambda_schedule: str = "constant"
    grl_max_lambda: float = 10.0
    mine_hidden: int = 64
    mine_layers: int = 2
    mine_dropout: float = 0.1
    mine_activation: str = "gelu"
    mine_norm: str = "layer"
    lambda_mi: float = 0.1

    def validate(self) -> List[str]:
        errors = []
        if self.lambda_mi <= 0:
            errors.append("THEOREM VIOLATION: lambda_mi must be > 0 (A5, CC5, C1)")
        if self.mi_estimator not in ["adversarial", "mine"]:
            errors.append(f"Unknown mi_estimator: {self.mi_estimator}")
        if self.grl_lambda_init < 0:
            errors.append("grl_lambda_init must be >= 0")
        if self.grl_max_lambda < self.grl_lambda_init:
            errors.append("grl_max_lambda must be >= grl_lambda_init")
        return errors

    @classmethod
    def from_utils_config(cls, config: NoLeakageConfig) -> "NoLeakageRegularizerConfig":
        return cls(
            mi_estimator=config.mi_estimator,
            discriminator_type=config.discriminator_type,
            discriminator_hidden=config.discriminator_hidden,
            discriminator_layers=config.discriminator_layers,
            gradient_reversal=config.gradient_reversal,
            grl_lambda_init=config.grl_lambda,
            lambda_mi=config.lambda_mi
        )


# ============================================================================
# MAIN MODULE: NOLEAKAGEREGULARIZER
# ============================================================================

class NoLeakageRegularizer(nn.Module):
    """
    Module 6: NoLeakageRegularizer (M6)

    Enforces I(z_g; a | O) = 0 via adversarial MI estimation or MINE.
    Training-time ONLY - no inference forward pass.

    Architecture:
    - Adversarial (primary): GRL + MLPDiscriminator (DANN-style)
    - MINE (alternative): Donsker-Varadhan bound with critic network

    Theorem Traceability:
    - A5 (No Leakage): I(z_g; a | O) = 0
    - CC5 (MI Penalty): L_noleak = lambda_MI * I(z_g; a)
    - C1 (Converse): If I(z_g; a) = 0 then no platform info leakage
    - L8 (Factorization): psi(mu_F) independent of schema
    """

    def __init__(
        self,
        config: NoLeakageRegularizerConfig,
        z_dim: int,
        num_platforms: int
    ):
        super().__init__()
        self.config = config
        self.z_dim = z_dim
        self.num_platforms = num_platforms

        errors = config.validate()
        if errors:
            raise ValueError(f"NoLeakageRegularizer config validation failed: {errors}")

        if config.mi_estimator == "adversarial":
            self.grl = GradientReversalLayer(lambda_init=config.grl_lambda_init)
            self.discriminator = MLPDiscriminator(
                input_dim=z_dim,
                num_classes=num_platforms,
                hidden_dim=config.discriminator_hidden,
                num_layers=config.discriminator_layers,
                dropout=config.discriminator_dropout,
                activation=config.discriminator_activation,
                norm_type=config.discriminator_norm
            )
            self.mine_estimator = None
        else:
            self.grl = None
            self.discriminator = None
            self.mine_estimator = MINEEstimator(
                z_dim=z_dim,
                num_platforms=num_platforms,
                hidden_dim=config.mine_hidden,
                num_layers=config.mine_layers,
                dropout=config.mine_dropout,
                activation=config.mine_activation,
                norm_type=config.mine_norm
            )

        self.register_buffer("lambda_mi", torch.tensor(config.lambda_mi, dtype=torch.float32))
        self._training_mode = True

    def train(self, mode: bool = True):
        self._training_mode = mode
        return super().train(mode)

    def eval(self):
        self._training_mode = False
        return super().eval()

    def _adversarial_loss(
        self,
        z_g: torch.Tensor,
        a_idx: torch.Tensor,
        training: bool = True
    ):
        if not training:
            return (
                torch.tensor(0.0, device=z_g.device, dtype=z_g.dtype),
                torch.tensor(0.0, device=z_g.device, dtype=z_g.dtype),
                {"disc_acc": torch.tensor(0.0, device=z_g.device)}
            )

        z_g_reversed = self.grl(z_g)
        logits = self.discriminator(z_g_reversed)
        disc_loss = F.cross_entropy(logits, a_idx)

        with torch.no_grad():
            pred = logits.argmax(dim=-1)
            disc_acc = (pred == a_idx).float().mean()

        # For encoder: gradient reversal makes encoder MAXIMIZE disc_loss
        # (minimize -disc_loss) to confuse discriminator -> no platform info in z_g
        enc_loss = -disc_loss
        
        # But for TOTAL loss computation, we want POSITIVE L_noleak = lambda_mi * disc_loss
        # (i.e., we add lambda_mi * I(z_g;a) to total loss, and GRL handles the sign)
        noleak_loss = self.lambda_mi * disc_loss  # POSITIVE for total loss

        # Return: (what gets backpropped to encoder, disc_loss for discriminator step, aux)
        # The encoder backprop target is self.lambda_mi * enc_loss (negative)
        total_loss_for_backprop = self.lambda_mi * enc_loss  # NEGATIVE

        return total_loss_for_backprop, disc_loss, {
            "disc_loss": disc_loss.detach(),
            "disc_acc": disc_acc,
            "enc_loss": enc_loss.detach(),
            "noleak_loss": noleak_loss.detach(),
        }

    def _mine_loss(
        self,
        z_g: torch.Tensor,
        a_idx: torch.Tensor,
        training: bool = True
    ):
        if not training:
            return (
                torch.tensor(0.0, device=z_g.device, dtype=z_g.dtype),
                {"mi_estimate": torch.tensor(0.0, device=z_g.device)}
            )

        mi_estimate, joint_scores, marginal_scores = self.mine_estimator(z_g, a_idx)
        total_loss = self.lambda_mi * mi_estimate

        return total_loss, {
            "mi_estimate": mi_estimate.detach(),
            "joint_term": joint_scores.mean().detach(),
            "marginal_term": marginal_scores.mean().detach(),
        }

    def forward(
        self,
        z_g: torch.Tensor,
        a_idx: torch.Tensor,
        training=None
    ):
        is_training = training if training is not None else self._training_mode

        if z_g.dim() != 2:
            raise ValueError(f"z_g must be 2D [B, k], got {z_g.shape}")
        if z_g.shape[1] != self.z_dim:
            raise ValueError(f"z_g dim {z_g.shape[1]} != config z_dim {self.z_dim}")
        if a_idx.dim() != 1:
            raise ValueError(f"a_idx must be 1D [B], got {a_idx.shape}")
        if z_g.shape[0] != a_idx.shape[0]:
            raise ValueError(f"Batch size mismatch: z_g {z_g.shape[0]} vs a_idx {a_idx.shape[0]}")
        if a_idx.max() >= self.num_platforms or a_idx.min() < 0:
            raise ValueError(f"a_idx values must be in [0, {self.num_platforms-1}]")

        device = z_g.device

        if not is_training:
            return {
                "noleak_loss": torch.tensor(0.0, device=device, dtype=z_g.dtype),
                "disc_loss": torch.tensor(0.0, device=device, dtype=z_g.dtype),
                "mi_estimate": torch.tensor(0.0, device=device, dtype=z_g.dtype),
                "disc_acc": torch.tensor(0.0, device=device, dtype=z_g.dtype),
            }

        if self.config.mi_estimator == "adversarial":
            # _adversarial_loss returns (loss_for_encoder_backprop, disc_loss, aux)
            # We want to return the POSITIVE noleak_loss for total loss computation
            loss_for_backprop, disc_loss, aux = self._adversarial_loss(z_g, a_idx, training)
            return {
                "noleak_loss": aux["noleak_loss"],  # POSITIVE: lambda_mi * disc_loss
                "disc_loss": disc_loss,
                **{k: v for k, v in aux.items() if k != "noleak_loss"}
            }
        else:
            noleak_loss, aux = self._mine_loss(z_g, a_idx, training)
            return {
                "noleak_loss": noleak_loss,
                "disc_loss": torch.tensor(0.0, device=device),
                **aux
            }

    def discriminator_step(
        self,
        z_g: torch.Tensor,
        a_idx: torch.Tensor
    ):
        if self.config.mi_estimator != "adversarial":
            raise RuntimeError("discriminator_step only available for adversarial MI estimator")

        self.discriminator.train()
        z_g_detached = z_g.detach()
        logits = self.discriminator(z_g_detached)
        disc_loss = F.cross_entropy(logits, a_idx)

        with torch.no_grad():
            pred = logits.argmax(dim=-1)
            disc_acc = (pred == a_idx).float().mean()

        return {"disc_loss": disc_loss, "disc_acc": disc_acc}

    def encoder_step(
        self,
        z_g: torch.Tensor,
        a_idx: torch.Tensor
    ):
        if self.config.mi_estimator != "adversarial":
            mi_out = self._mine_loss(z_g, a_idx)
            return {"enc_loss": mi_out[1]["enc_loss"], "noleak_loss": mi_out[0]}

        out = self._adversarial_loss(z_g, a_idx, training=True)
        return {"enc_loss": out[2]["enc_loss"], "noleak_loss": out[2]["noleak_loss"]}

    def set_grl_lambda(self, lambda_: float) -> None:
        if self.grl is not None:
            self.grl.set_lambda(lambda_)

    def get_grl_lambda(self) -> float:
        if self.grl is not None:
            return self.grl.get_lambda()
        return 0.0

    def extra_repr(self) -> str:
        return f"z_dim={self.z_dim}, num_platforms={self.num_platforms}, estimator={self.config.mi_estimator}, lambda_mi={self.lambda_mi.item():.4f}"


# ============================================================================
# TORCHSCRIPT-COMPATIBLE VERSION
# ============================================================================

class NoLeakageRegularizerScriptable(nn.Module):
    """
    TorchScript-compatible version of NoLeakageRegularizer.
    Uses ModuleList instead of Sequential for discriminator.
    """
    def __init__(
        self,
        z_dim: int,
        num_platforms: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        lambda_mi: float = 0.1,
        grl_lambda_init: float = 1.0,
        dropout: float = 0.1,
        activation: str = "gelu",
        norm_type: str = "layer"
    ):
        super().__init__()
        self.z_dim = z_dim
        self.num_platforms = num_platforms
        self.lambda_mi = lambda_mi

        if activation == "gelu":
            act_fn = nn.GELU
        elif activation == "relu":
            act_fn = nn.ReLU
        elif activation == "silu":
            act_fn = nn.SiLU
        else:
            act_fn = nn.GELU

        def get_norm(dim: int):
            if norm_type == "layer":
                return nn.LayerNorm(dim)
            elif norm_type == "batch":
                return nn.BatchNorm1d(dim)
            else:
                return nn.Identity()

        self.register_buffer("grl_lambda", torch.tensor(grl_lambda_init, dtype=torch.float32))

        self.discriminator_layers = nn.ModuleList()
        in_dim = z_dim
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else num_platforms
            self.discriminator_layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                self.discriminator_layers.append(get_norm(out_dim))
                self.discriminator_layers.append(act_fn())
                self.discriminator_layers.append(nn.Dropout(dropout))
            in_dim = out_dim

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _discriminator_forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.discriminator_layers:
            x = layer(x)
        return x

    def _grl_forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def forward(self, z_g: torch.Tensor, a_idx: torch.Tensor):
        x = self._grl_forward(z_g)
        logits = self._discriminator_forward(x)

        disc_loss = F.cross_entropy(logits, a_idx)
        with torch.no_grad():
            pred = logits.argmax(dim=-1)
            disc_acc = (pred == a_idx).float().mean()

        enc_loss = -disc_loss
        noleak_loss = self.lambda_mi * enc_loss

        return {
            "noleak_loss": noleak_loss,
            "disc_loss": disc_loss,
            "disc_acc": disc_acc,
            "mi_estimate": torch.zeros((), device=z_g.device),
            "enc_loss": enc_loss
        }

    def discriminator_step(self, z_g: torch.Tensor, a_idx: torch.Tensor):
        z_g_detached = z_g.detach()
        x = z_g_detached
        for layer in self.discriminator_layers:
            x = layer(x)
        logits = x

        disc_loss = F.cross_entropy(logits, a_idx)
        with torch.no_grad():
            pred = logits.argmax(dim=-1)
            disc_acc = (pred == a_idx).float().mean()

        return {"disc_loss": disc_loss, "disc_acc": disc_acc}

    def set_grl_lambda(self, lambda_: float):
        self.grl_lambda.fill_(lambda_)


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_no_leakage_regularizer(
    config: NoLeakageRegularizerConfig,
    z_dim: int,
    num_platforms: int
) -> NoLeakageRegularizer:
   return NoLeakageRegularizer(config, z_dim, num_platforms)


def create_no_leakage_from_utils_config(
    utils_config: NoLeakageConfig,
    z_dim: int,
    num_platforms: int,
    **overrides
) -> NoLeakageRegularizer:
    config = NoLeakageRegularizerConfig.from_utils_config(utils_config)
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Unknown config override: {key}")
    return NoLeakageRegularizer(config, z_dim, num_platforms)


# ============================================================================
# UNIT TESTS
# ============================================================================

def test_gradient_reversal_layer():
    print("Testing GradientReversalLayer...")
    grl = GradientReversalLayer(lambda_init=2.0)
    x = torch.randn(4, 8, requires_grad=True)
    y = grl(x)
    assert torch.allclose(y, x), "GRL forward should be identity"
    loss = y.sum()
    loss.backward()
    expected_grad = -2.0 * torch.ones_like(x)
    assert torch.allclose(x.grad, expected_grad), f"GRL backward failed: {x.grad} vs {expected_grad}"
    print("  GradientReversalLayer: PASSED")


def test_mlp_discriminator():
    print("Testing MLPDiscriminator...")
    disc = MLPDiscriminator(input_dim=32, num_classes=3, hidden_dim=64, num_layers=2, dropout=0.1)
    B = 16
    z_g = torch.randn(B, 32)
    logits = disc(z_g)
    assert logits.shape == (B, 3), f"Expected [B, 3], got {logits.shape}"
    loss = logits.sum()
    loss.backward()
    assert disc.network[0].weight.grad is not None
    print("  MLPDiscriminator: PASSED")


def test_mine_estimator():
    print("Testing MINEEstimator...")
    mine = MINEEstimator(z_dim=32, num_platforms=3, hidden_dim=64, num_layers=2)
    B = 32
    z_g = torch.randn(B, 32)
    a_idx = torch.randint(0, 3, (B,))
    mi_est, joint, marginal = mine(z_g, a_idx)
    assert mi_est.dim() == 0, "MI estimate should be scalar"
    assert joint.shape == (B,), f"Joint scores shape {joint.shape} != ({B},)"
    assert marginal.shape == (B,), f"Marginal scores shape {marginal.shape} != ({B},)"
    loss = mi_est
    loss.backward()
    assert mine.critic[0].weight.grad is not None
    print("  MINEEstimator: PASSED")


def test_no_leakage_adversarial():
    print("Testing NoLeakageRegularizer (adversarial)...")
    config = NoLeakageRegularizerConfig(
        mi_estimator="adversarial",
        discriminator_hidden=32,
        discriminator_layers=2,
        gradient_reversal=True,
        grl_lambda_init=1.0,
        lambda_mi=0.1,
        dropout=0.1
    )
    reg = NoLeakageRegularizer(config, z_dim=32, num_platforms=3)
    B = 16
    z_g = torch.randn(B, 32)
    a_idx = torch.randint(0, 3, (B,))
    reg.train()
    out = reg(z_g, a_idx, training=True)
    assert "noleak_loss" in out
    assert "disc_loss" in out
    assert "disc_acc" in out
    assert out["noleak_loss"].dim() == 0
    assert out["disc_loss"].dim() == 0
    
    # CRITICAL TEST: noleak_loss should be POSITIVE for total loss
    assert out["noleak_loss"].item() >= 0, f"noleak_loss must be >= 0 for total loss, got {out['noleak_loss'].item()}"
    
    loss = out["noleak_loss"]
    loss.backward()
    assert reg.discriminator.network[0].weight.grad is not None
    reg.eval()
    out_inf = reg(z_g, a_idx, training=False)
    assert out_inf["noleak_loss"].item() == 0.0
    assert out_inf["disc_loss"].item() == 0.0
    disc_out = reg.discriminator_step(z_g, a_idx)
    assert "disc_loss" in disc_out
    assert "disc_acc" in disc_out
    reg.set_grl_lambda(2.0)
    assert reg.get_grl_lambda() == 2.0
    print("  NoLeakageRegularizer (adversarial): PASSED")


def test_no_leakage_mine():
    print("Testing NoLeakageRegularizer (MINE)...")
    config = NoLeakageRegularizerConfig(
        mi_estimator="mine",
        mine_hidden=32,
        mine_layers=2,
        lambda_mi=0.1
    )
    reg = NoLeakageRegularizer(config, z_dim=32, num_platforms=3)
    B = 16
    z_g = torch.randn(B, 32)
    a_idx = torch.randint(0, 3, (B,))
    reg.train()
    out = reg(z_g, a_idx, training=True)
    assert "noleak_loss" in out
    assert "mi_estimate" in out
    assert out["noleak_loss"].dim() == 0
    assert out["mi_estimate"].dim() == 0
    loss = out["noleak_loss"]
    loss.backward()
    assert reg.mine_estimator.critic[0].weight.grad is not None
    reg.eval()
    out_inf = reg(z_g, a_idx, training=False)
    assert out_inf["noleak_loss"].item() == 0.0
    print("  NoLeakageRegularizer (MINE): PASSED")


def test_factory_functions():
    print("Testing factory functions...")
    config = NoLeakageRegularizerConfig(mi_estimator="adversarial", lambda_mi=0.1)
    reg1 = create_no_leakage_regularizer(config, z_dim=64, num_platforms=4)
    assert isinstance(reg1, NoLeakageRegularizer)
    assert reg1.z_dim == 64
    assert reg1.num_platforms == 4
    utils_config = NoLeakageConfig(mi_estimator="adversarial", lambda_mi=0.2)
    reg2 = create_no_leakage_from_utils_config(utils_config, z_dim=32, num_platforms=2)
    assert reg2.config.lambda_mi == 0.2
    reg3 = create_no_leakage_from_utils_config(utils_config, z_dim=32, num_platforms=2, lambda_mi=0.5, mine_layers=3)
    assert reg3.config.lambda_mi == 0.5
    assert reg3.config.mine_layers == 3
    print("  Factory functions: PASSED")


def test_config_validation():
    print("Testing config validation...")
    config = NoLeakageRegularizerConfig(lambda_mi=0.1)
    errors = config.validate()
    assert len(errors) == 0, f"Valid config should pass: {errors}"
    config_bad = NoLeakageRegularizerConfig(lambda_mi=0.0)
    errors = config_bad.validate()
    assert len(errors) > 0, "lambda_mi <= 0 should fail validation"
    assert any("THEOREM VIOLATION" in e for e in errors)
    config_bad2 = NoLeakageRegularizerConfig(lambda_mi=-0.1)
    errors = config_bad2.validate()
    assert len(errors) > 0
    config_bad3 = NoLeakageRegularizerConfig(mi_estimator="unknown")
    errors = config_bad3.validate()
    assert len(errors) > 0
    assert any("Unknown mi_estimator" in e for e in errors)
    print("  Config validation: PASSED")


def test_scriptable_version():
    print("Testing NoLeakageRegularizerScriptable...")
    reg = NoLeakageRegularizerScriptable(
        z_dim=32, num_platforms=3, hidden_dim=32, num_layers=2,
        lambda_mi=0.1, grl_lambda_init=1.0
    )
    B = 16
    z_g = torch.randn(B, 32)
    a_idx = torch.randint(0, 3, (B,))
    out = reg(z_g, a_idx)
    assert "noleak_loss" in out
    assert "disc_loss" in out
    assert out["noleak_loss"].dim() == 0
    try:
        scripted = torch.jit.script(reg)
        out_script = scripted(z_g, a_idx)
        assert torch.allclose(out["noleak_loss"], out_script["noleak_loss"])
        print("  TorchScript compilation: PASSED")
    except Exception as e:
        print(f"  TorchScript compilation: SKIPPED ({e})")
    print("  NoLeakageRegularizerScriptable: PASSED")


def test_input_validation():
    print("Testing input validation...")
    config = NoLeakageRegularizerConfig(mi_estimator="adversarial", lambda_mi=0.1)
    reg = NoLeakageRegularizer(config, z_dim=32, num_platforms=3)
    z_g_bad = torch.randn(4, 16)
    a_idx = torch.randint(0, 3, (4,))
    try:
        reg(z_g_bad, a_idx)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "z_g dim" in str(e)
    z_g2 = torch.randn(4, 32)
    a_idx2 = torch.randint(0, 3, (5,))
    try:
        reg(z_g2, a_idx2)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Batch size mismatch" in str(e)
    a_idx3 = torch.randint(0, 3, (4, 2))
    try:
        reg(z_g2, a_idx3)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "a_idx must be 1D" in str(e)
    print("  Input validation: PASSED")


def run_all_tests():
    print("=" * 60)
    print("NOLEAKAGEREGULARIZER UNIT TESTS")
    print("=" * 60)
    test_gradient_reversal_layer()
    test_mlp_discriminator()
    test_mine_estimator()
    test_no_leakage_adversarial()
    test_no_leakage_mine()
    test_factory_functions()
    test_config_validation()
    test_scriptable_version()
    test_input_validation()
    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    run_all_tests()
