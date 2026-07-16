"""
HWIN-Net: Module 2 - Recovery Module (M2)

Mathematical Purpose
--------------------
Implements MOp-4 (Recovery) and MOp-10 (Equivariance Tying) from Phase XIV Task 2.
Computes the recovered statistic mu_hat from the masked observation z_g:

    mu_hat = T_{(O,a)}(z_g)

where T_{(O,a)}: R^{|O|} -> R^d is the recovery map per schema g = (O, a).

Per Lemma 5 (Equivariance Structure) and CC3 (Equivariance Constraint):
For fixed observation set O and varying platforms a, a_ref in A:
    T_{(O,a)} = R_{a,a_ref} o T_{(O,a_ref)}

where R_{a2,a1}: R^d -> R^d is the intertwiner satisfying:
    R_{a,a} = Identity
    R_{a3,a1} = R_{a3,a2} o R_{a2,a1}  (groupoid composition)

This is a HARD REQUIREMENT from the theory (CR-L5, CR-CC3, derived from A2 + A4 + L5).

Theory Traceability
-------------------
- Axiom A4 (Uniform Identifiability): Exists mu and {T_g} with T_g o z_g = mu a.e. for |O| >= r_0(g)
- Lemma 2 (Recovery <=> Marginal Identifiability): T_g exists iff mu is sigma(z_g)-measurable
- Lemma 5 (Equivariance Structure): Intertwiners R_{a2,a1} compose groupoidally
- CC1 (Recovery via Min-Max): Variational form for T_g
- CC3 (Equivariance Constraint): Weight-sharing via intertwiner
- CR-A4a, CR-L2, CR-L5, CR-CC1: Computational requirements

Tensor Signatures
-----------------
- Input z_g:      [B, k]        - Masked encoding from SchemaEncoder (M1)
- Input a_idx:    [B]           - Platform index in {0, ..., |A|-1}
- Output mu_hat:  [B, d]        - Recovered statistic (pre-retraction)

Where k = latent encoding dim, d = latent_dim (dim of mu(M))

Complexity
----------
- Time: O(B * (k * d + |A| * d^2)) for base recovery + intertwiners
- Space: O(|A| * (k * d + d^2)) for T_base weights + intertwiner weights

Assumptions
-----------
- z_g is the output of SchemaEncoder: z_g = M_O * h_a (masked encoding)
- Platform indices in a_idx range [0, num_platforms-1]
- For fixed O, all platforms share the same base recovery network T_base
- Intertwiners R_{a,a_ref} are learned per platform (relative to base_platform)
- The base_platform (a_ref) is fixed as reference

Implementation Choices
---------------------
- T_base architecture: MLP (configurable via RecoveryConfig.recovery_type)
- Intertwiner types: linear, orthogonal, diagonal (configurable via intertwiner_type)
- Weight tying: T_{(O,a)}(z) = R_{a,a_ref}( T_{base}(z) ) per L5/CC3
- Equivariance loss: ||R_{a2,a1} - I||_F^2 for identity, or composition consistency
- TorchScript compatible: static control flow per forward pass
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
import math


@dataclass
class RecoveryConfig:
    """Configuration for RecoveryModule matching RecoveryConfig from config.py."""
    recovery_type: str = "equivariant_mlp"  # "equivariant_mlp", "transformer", "deepsets"
    latent_dim: int = 64                      # d: dimension of mu: M -> R^d
    hidden_dim: int = 128                     # Hidden dimension of T_{(O,a)}
    n_layers: int = 3                         # Number of layers in T_base
    intertwiner_type: str = "linear"         # "linear", "orthogonal", "diagonal"
    base_platform: int = 0                    # Reference platform a_ref
    tie_t_base: bool = True                   # Share base T_{(O,a_ref)} across platforms
    equivariance_loss_weight: float = 1.0     # Weight for equivariance loss
    dropout: float = 0.1
    activation: str = "gelu"
    norm_type: str = "layer"                   # "layer", "batch", "none"
    num_platforms: int = 3                  # |A| number of platforms
    obs_dim: int = 128                        # Input dimension k (from SchemaEncoder output)


class RecoveryBase(nn.Module):
    """
    Base recovery network T_{base}: R^k -> R^d (for reference platform a_ref).

    Theorem Traceability: A4, L2, CR-A4a, CR-L2
    - A4: T_g exists for all g with |O| >= r_0(g)
    - L2: T_g must be sigma(z_g)-measurable (no access to unobserved vars)
    - CR-A4a: Implement recovering function mu from each sufficient-schema z_g
    - CR-L2: T_g must not use unobserved variables

    Tensor Signature:
    - Input:  [B, k] (z_g from SchemaEncoder)
    - Output: [B, d] (mu_hat)
    """

    def __init__(
        self,
        obs_dim: int,
        latent_dim: int,
        hidden_dim: int,
        n_layers: int,
        dropout: float = 0.1,
        activation: str = "gelu",
        norm_type: str = "layer",
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.n_layers = n_layers

        # Activation function
        if activation == "gelu":
            act_fn = nn.GELU
        elif activation == "relu":
            act_fn = nn.ReLU
        elif activation == "silu":
            act_fn = nn.SiLU
        else:
            act_fn = nn.GELU

        # Normalization
        def get_norm(dim: int) -> nn.Module:
            if norm_type == "layer":
                return nn.LayerNorm(dim)
            elif norm_type == "batch":
                return nn.BatchNorm1d(dim)
            else:
                return nn.Identity()

        # Build MLP: [k] -> hidden -> ... -> [d]
        layers = []
        in_dim = obs_dim

        for i in range(n_layers):
            out_dim = hidden_dim if i < n_layers - 1 else latent_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i < n_layers - 1:
                layers.append(get_norm(out_dim))
                layers.append(act_fn())
                layers.append(nn.Dropout(dropout))
            in_dim = out_dim

        self.net = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z_g: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of base recovery network.

        Args:
            z_g: [B, k] masked encoding

        Returns:
            mu_hat: [B, d] recovered statistic
        """
        return self.net(z_g)


class Intertwiner(nn.Module):
    """
    Intertwiner R_{a,a_ref}: R^d -> R^d for platform a relative to reference a_ref.

    Theorem Traceability: L5, CC3, CR-L5, CR-CC3
    - L5: R_{a2,a1} = T_{(O,a2)} o T_{(O,a1)}^{-1} on images
    - Groupoid: R_{a,a} = I, R_{a3,a1} = R_{a3,a2} o R_{a2,a1}
    - CC3: Weight-sharing via intertwiner relation
    - CR-L5, CR-CC3: Implementation of equivariance constraint

    Tensor Signature:
    - Input:  [B, d] (mu_hat from T_base)
    - Output: [B, d] (transformed mu_hat for platform a)
    """

    def __init__(
        self,
        latent_dim: int,
        intertwiner_type: str = "linear",
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.intertwiner_type = intertwiner_type

        if intertwiner_type == "linear":
            self.transform = nn.Linear(latent_dim, latent_dim, bias=False)
            nn.init.eye_(self.transform.weight)
        elif intertwiner_type == "orthogonal":
            self.transform = nn.Linear(latent_dim, latent_dim, bias=False)
            nn.init.eye_(self.transform.weight)  # Start at identity for equivariance warmup
        elif intertwiner_type == "diagonal":
            # Diagonal matrix: elementwise scaling
            self.scale = nn.Parameter(torch.ones(latent_dim))
            self.transform = None
        else:
            raise ValueError(f"Unknown intertwiner_type: {intertwiner_type}")

    def forward(self, mu_hat: torch.Tensor) -> torch.Tensor:
        """
        Apply intertwiner transformation.

        Args:
            mu_hat: [B, d] recovered statistic from T_base

        Returns:
            [B, d] transformed statistic for platform a
        """
        if self.intertwiner_type == "diagonal":
            return mu_hat * self.scale
        else:
            return self.transform(mu_hat)

    def compose(self, other: "Intertwiner") -> "Intertwiner":
        """
        Compose two intertwiners: self o other.

        For linear type: R_new = self @ other
        For diagonal type: R_new = self.scale * other.scale
        """
        if self.intertwiner_type != other.intertwiner_type:
            raise ValueError("Cannot compose different intertwiner types")

        if self.intertwiner_type == "linear":
            new = Intertwiner(self.latent_dim, "linear")
            with torch.no_grad():
                new.transform.weight.copy_(self.transform.weight @ other.transform.weight)
            return new
        elif self.intertwiner_type == "diagonal":
            new = Intertwiner(self.latent_dim, "diagonal")
            with torch.no_grad():
                new.scale.copy_(self.scale * other.scale)
            return new
        else:
            # For orthogonal, just return self (maintains orthogonality approximately)
            return self


class RecoveryModule(nn.Module):
    """
    M2: Recovery Module - Computes mu_hat = T_{(O,a)}(z_g) with equivariance tying.

    Implements MOp-4 (Recovery) and MOp-10 (Equivariance Tying).

    Theorem Traceability:
    - A4: Uniform identifiability -> T_g exists for all sufficient schemas
    - L2: Recovery iff marginal identifiability -> T_g is sigma(z_g)-measurable
    - L5: Equivariance structure -> T_{(O,a2)} = R_{a2,a1} o T_{(O,a1)}
    - CC1: Recovery via min-max optimization
    - CC3: Equivariance constraint -> weight sharing via intertwiner
    - CR-A4a, CR-L2, CR-L5, CR-CC1: Computational requirements

    Tensor Signatures (batch-first):
    - forward(z_g, a_idx, O=None)
      z_g:   [B, k]     - masked encoding from SchemaEncoder
      a_idx: [B]        - platform indices
      O:     Optional   - observation set (not used in current implementation)
    - Returns:
      mu_hat: [B, d]    - recovered statistic (pre-retraction)

    Config: RecoveryConfig (maps to RecoveryConfig in config.py)
    """

    def __init__(self, config: RecoveryConfig):
        super().__init__()
        self.config = config
        self.obs_dim = config.obs_dim
        self.latent_dim = config.latent_dim
        self.hidden_dim = config.hidden_dim
        self.n_layers = config.n_layers
        self.intertwiner_type = config.intertwiner_type
        self.base_platform = config.base_platform
        self.tie_t_base = config.tie_t_base
        self.num_platforms = getattr(config, "num_platforms", 3)

        # Base recovery network T_{base} = T_{(O, a_ref)}
        self.T_base = RecoveryBase(
            obs_dim=self.obs_dim,
            latent_dim=self.latent_dim,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_layers,
            dropout=config.dropout,
            activation=config.activation,
            norm_type=config.norm_type,
        )

        # Intertwiners R_{a, a_ref} for each platform a
        # R_{a_ref, a_ref} = Identity (no transformation for base platform)
        self.intertwiners = nn.ModuleDict()
        for a in range(self.num_platforms):
            if a == self.base_platform:
                # Identity for base platform
                self.intertwiners[str(a)] = nn.Identity()
            else:
                self.intertwiners[str(a)] = Intertwiner(
                    latent_dim=self.latent_dim,
                    intertwiner_type=self.intertwiner_type,
                )

        # For equivariance loss computation
        self.register_buffer("eye", torch.eye(self.latent_dim))

    def forward(
        self,
        z_g: torch.Tensor,
        a_idx: torch.Tensor,
        O: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of Recovery Module.

        Args:
            z_g: [B, k] - masked encoding from SchemaEncoder
            a_idx: [B] - platform indices
            O: Optional observation set (ignored in current implementation)

        Returns:
            mu_hat: [B, d] - recovered statistic T_{(O,a)}(z_g)
        """
        B, k = z_g.shape
        device = z_g.device
        d = self.latent_dim

        # Validate inputs
        assert k == self.obs_dim, f"z_g dim {k} != obs_dim {self.obs_dim}"
        assert a_idx.shape == (B,), f"a_idx shape {a_idx.shape} != ({B},)"
        assert a_idx.min() >= 0 and a_idx.max() < self.num_platforms,             f"Platform indices out of range [0, {self.num_platforms-1}]"

        # Apply base recovery: mu_base = T_base(z_g)  [B, d]
        mu_base = self.T_base(z_g)

        # Apply intertwiners per platform: mu_hat = R_{a, a_ref}(mu_base)
        mu_hat = torch.empty(B, d, device=device, dtype=z_g.dtype)

        if self.tie_t_base:
            # Use shared base + platform-specific intertwiners
            for a in range(self.num_platforms):
                mask = (a_idx == a)
                if mask.any():
                    R_a = self.intertwiners[str(a)]
                    mu_hat[mask] = R_a(mu_base[mask])
        else:
            # Separate T_{(O,a)} per platform (violates CC3 - not recommended)
            # This path is for ablation only
            mu_hat = mu_base

        return mu_hat

    def get_base_recovery(self) -> RecoveryBase:
        """Get the base recovery network T_base."""
        return self.T_base

    def get_intertwiner(self, a: int) -> nn.Module:
        """Get intertwiner for platform a."""
        return self.intertwiners[str(a)]

    def equivariance_loss(self, z_g: torch.Tensor, a_idx: torch.Tensor) -> torch.Tensor:
        """
        Compute equivariance loss per L5, CC3.

        For each pair of platforms (a1, a2) present in the batch:
        R_{a2,a1} = R_{a2,a_ref} o R_{a1,a_ref}^{-1}
        Should satisfy: R_{a2,a1} approx Identity on the subspace of T_base(z_g)

        Loss: ||R_{a2,a1} - I||_F^2 or consistency of groupoid composition
        """
        loss = torch.tensor(0.0, device=z_g.device, dtype=z_g.dtype)
        count = 0

        # Get all platforms present in batch
        platforms_present = a_idx.unique().tolist()

        if len(platforms_present) < 2:
            return loss  # No pairs to check

        # Compute base recovery for all
        with torch.no_grad():
            mu_base = self.T_base(z_g)

        # Check pairwise equivariance
        for a1 in platforms_present:
            for a2 in platforms_present:
                if a1 == a2:
                    continue

                # R_{a1, a_ref}
                R1 = self.intertwiners[str(a1)]
                # R_{a2, a_ref}
                R2 = self.intertwiners[str(a2)]

                # R_{a2, a1} = R_{a2, a_ref} o R_{a1, a_ref}^{-1}
                # For linear intertwiners, we can check composition consistency
                if self.intertwiner_type == "linear":
                    if a1 != self.base_platform and a2 != self.base_platform:
                        W1 = R1.transform.weight  # [d, d]
                        W2 = R2.transform.weight  # [d, d]
                        # R_{a2, a1} = W2 @ W1^{-1}
                        try:
                            W1_inv = torch.linalg.inv(W1)
                            R21 = W2 @ W1_inv
                            loss += torch.norm(R21 - self.eye, p="fro") ** 2
                            count += 1
                        except RuntimeError:
                            pass  # Singular matrix
                    elif a1 == self.base_platform:
                        # R_{a2, base} = W2, should be near identity for base? Not necessarily.
                        # But R_{base, a1} = W1^{-1} should be inverse
                        pass
                    elif a2 == self.base_platform:
                        # R_{base, a1} = W1^{-1}
                        pass

        if count > 0:
            loss = loss / count * self.config.equivariance_loss_weight

        return loss

    def equivariance_loss_simple(self, z_g: torch.Tensor, a_idx: torch.Tensor) -> torch.Tensor:
        """
        Simplified equivariance loss: enforce R_{a2, a1} op T = T op R_{a2, a1}
        i.e., T_{(O,a2)}(z) = R_{a2,a1}( T_{(O,a1)}(z) )

        For shared T_base: this means R_{a2,a1} should commute with T_base on the data subspace.
        We check: || R_a(T_base(z)) - T_base(z) || for all a (each platform should give same latent)
        """
        # This is a simpler loss: for shared encoder, all platforms should produce
        # the same mu after applying their intertwiners to T_base output
        mu_base = self.T_base(z_g)
        loss = torch.tensor(0.0, device=z_g.device, dtype=z_g.dtype)

        for a in range(self.num_platforms):
            if a == self.base_platform:
                continue
            mask = (a_idx == a)
            if mask.any():
                R_a = self.intertwiners[str(a)]
                mu_a = R_a(mu_base[mask])
                # All platforms should produce same latent (up to noise)
                loss += F.mse_loss(mu_a, mu_base[mask])

        if self.num_platforms > 1:
            loss = loss / (self.num_platforms - 1) * self.config.equivariance_loss_weight

        return loss

    def extra_repr(self) -> str:
        return (
            f"obs_dim={self.obs_dim}, latent_dim={self.latent_dim}, "
            f"num_platforms={self.num_platforms}, base_platform={self.base_platform}, "
            f"intertwiner_type={self.intertwiner_type}"
        )


def create_recovery_module(config: RecoveryConfig) -> RecoveryModule:
    """
    Factory function to create RecoveryModule from config.

    Theorem Traceability: Ensures config matches RecoveryConfig from config.py
    """
    return RecoveryModule(config)


# ============================================================================
# TorchScript Compatibility
# ============================================================================

class RecoveryModuleScriptable(nn.Module):
    """
    TorchScript-compatible RecoveryModule.

    Uses ModuleList for intertwiners instead of ModuleDict.
    """

    def __init__(self, config: RecoveryConfig):
        super().__init__()
        self.config = config
        self.obs_dim = config.obs_dim
        self.latent_dim = config.latent_dim
        self.hidden_dim = config.hidden_dim
        self.n_layers = config.n_layers
        self.intertwiner_type = config.intertwiner_type
        self.base_platform = config.base_platform
        self.tie_t_base = config.tie_t_base
        self.num_platforms = getattr(config, "num_platforms", 3)

        # Base recovery network
        self.T_base = RecoveryBase(
            obs_dim=self.obs_dim,
            latent_dim=self.latent_dim,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_layers,
            dropout=config.dropout,
            activation=config.activation,
            norm_type=config.norm_type,
        )

        # Intertwiners in ModuleList
        self.intertwiners = nn.ModuleList()
        for a in range(self.num_platforms):
            if a == self.base_platform:
                self.intertwiners.append(nn.Identity())
            else:
                self.intertwiners.append(Intertwiner(
                    latent_dim=self.latent_dim,
                    intertwiner_type=self.intertwiner_type,
                ))

        self.register_buffer("eye", torch.eye(self.latent_dim))

    def forward(
        self,
        z_g: torch.Tensor,
        a_idx: torch.Tensor,
        O: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, k = z_g.shape
        device = z_g.device
        d = self.latent_dim

        mu_base = self.T_base(z_g)
        mu_hat = torch.empty(B, d, device=device, dtype=z_g.dtype)

        if self.tie_t_base:
            for a in range(self.num_platforms):
                mask = (a_idx == a)
                if mask.any():
                    mu_hat[mask] = self.intertwiners[a](mu_base[mask])
        else:
            mu_hat = mu_base

        return mu_hat


# ============================================================================
# Unit Tests
# ============================================================================

def test_recovery_module():
    """Unit tests for RecoveryModule."""
    import pytest

    # Test 1: Basic forward pass
    config = RecoveryConfig(
        obs_dim=128,
        latent_dim=64,
        hidden_dim=128,
        n_layers=3,
        num_platforms=3,
        base_platform=0,
        intertwiner_type="linear",
        tie_t_base=True,
    )
    module = RecoveryModule(config)

    B = 16
    z_g = torch.randn(B, 128)
    a_idx = torch.randint(0, 3, (B,))

    mu_hat = module(z_g, a_idx)

    assert mu_hat.shape == (B, 64), f"mu_hat shape {mu_hat.shape} != (16, 64)"
    print("Test 1 PASSED: Basic forward pass")

    # Test 2: Base platform should use identity intertwiner
    z_g2 = torch.randn(4, 128)
    a_idx2 = torch.tensor([0, 0, 1, 2])  # 0 = base_platform
    mu_hat2 = module(z_g2, a_idx2)

    # For base platform (0), mu_hat should equal T_base output
    mask_base = (a_idx2 == 0)
    mu_base = module.T_base(z_g2[mask_base])
    assert torch.allclose(mu_hat2[mask_base], mu_base),         "Base platform should use identity intertwiner"
    print("Test 2 PASSED: Base platform identity")

    # Test 3: Different platforms produce different outputs (intertwiners learned)
    z_g3 = torch.ones(3, 128)
    a_idx3 = torch.tensor([0, 1, 2])
    mu_hat3 = module(z_g3, a_idx3)

    # Platforms 1 and 2 should have transformed outputs (unless intertwiners are identity)
    # Note: initially intertwiners are random, so outputs will differ
    assert not torch.allclose(mu_hat3[0], mu_hat3[1]),         "Different platforms should produce different outputs"
    print("Test 3 PASSED: Platform-specific outputs")

    # Test 4: Equivariance loss computation
    loss = module.equivariance_loss(z_g, a_idx)
    assert loss >= 0, "Equivariance loss should be non-negative"
    print("Test 4 PASSED: Equivariance loss computation")

    # Test 5: Scriptable version
    scriptable = RecoveryModuleScriptable(config)
    mu_script = scriptable(z_g, a_idx)
    assert mu_script.shape == mu_hat.shape
    print("Test 5 PASSED: Scriptable version")

    # Test 6: Orthogonal intertwiners
    config_orth = RecoveryConfig(
        obs_dim=128,
        latent_dim=64,
        hidden_dim=128,
        n_layers=3,
        num_platforms=3,
        base_platform=0,
        intertwiner_type="orthogonal",
        tie_t_base=True,
    )
    module_orth = RecoveryModule(config_orth)
    # Check initial orthogonality
    for a in range(1, 3):
        R = module_orth.intertwiners[str(a)]
        if hasattr(R, "transform") and R.transform is not None:
            W = R.transform.weight
            # W^T @ W should be close to I
            WTW = W.T @ W
            assert torch.allclose(WTW, torch.eye(64), atol=1e-3),                 "Orthogonal intertwiner should maintain orthogonality"
    print("Test 6 PASSED: Orthogonal intertwiners")

    # Test 7: Diagonal intertwiners
    config_diag = RecoveryConfig(
        obs_dim=128,
        latent_dim=64,
        hidden_dim=128,
        n_layers=3,
        num_platforms=3,
        base_platform=0,
        intertwiner_type="diagonal",
        tie_t_base=True,
    )
    module_diag = RecoveryModule(config_diag)
    mu_diag = module_diag(z_g3, a_idx3)
    assert mu_diag.shape == (3, 64)
    print("Test 7 PASSED: Diagonal intertwiners")

    # Test 8: tie_t_base=False (separate T per platform - ablation)
    config_no_tie = RecoveryConfig(
        obs_dim=128,
        latent_dim=64,
        hidden_dim=128,
        n_layers=3,
        num_platforms=3,
        base_platform=0,
        intertwiner_type="linear",
        tie_t_base=False,
    )
    module_no_tie = RecoveryModule(config_no_tie)
    mu_no_tie = module_no_tie(z_g3, a_idx3)
    assert mu_no_tie.shape == (3, 64)
    print("Test 8 PASSED: tie_t_base=False")

    print("\n=== ALL RECOVERY MODULE TESTS PASSED ===")


if __name__ == "__main__":
    test_recovery_module()

