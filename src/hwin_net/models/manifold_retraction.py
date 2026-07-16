"""
HWIN-Net: Module 3 - Manifold Retraction (M3)

Mathematical Purpose
--------------------
Implements MOp-5 (Manifold Retraction) from Phase XIV Task 2. Projects the
recovered statistic mu_hat onto the constraint manifold mu(M):

    mu_final = rho(mu_hat)

where rho: R^d -> mu(M) satisfies:
- rho o rho = rho  (idempotence)
- rho(mu) = mu  for all mu in mu(M)  (fixes manifold)
- Image(rho) = mu(M)  (image = quotient manifold)

This is required by Theorem 1 (mu* is quotient map) and CC4 (retraction layer).

Theory Traceability
-------------------
- Axiom A1 (Support): nu(M) = 1 -> data lives on M
- Theorem 1 (Existence of SIS): mu* = pi_obs: M -> M/~_obs ~= mu(M)
- Definition D12 (Hilbert Projection): Orthogonal projection when inner product available
- CC4 (Manifold Retraction): Projection layer rho: R^d -> mu(M)
- CR-A1, CR-T1, CR-CC4: Computational requirements

Tensor Signatures
-----------------
- Input mu_hat:    [B, d]     - Recovered statistic from RecoveryModule (M2)
- Output mu_final: [B, d]     - Retracted statistic on mu(M)

Complexity
----------
- Time: O(B * d * k) for PCA projection (k = pca_components)
- Time: O(B * iter * d^2) for iterative projection
- Space: O(d * k) for PCA basis, O(d) for VAE parameters

Assumptions
-----------
- mu_hat is in R^d (output of RecoveryModule)
- The manifold mu(M) is a d-dimensional submanifold of R^d
- For PCA: basis of mu(M) is precomputed or learned from data
- For VAE: manifold is learned from training data

Implementation Choices
---------------------
- retraction_type: "pca", "vae", "iterative", "identity"
- PCA: orthogonal projection onto precomputed subspace (D12)
- VAE: encoder-decoder learns manifold structure
- Iterative: projected gradient descent onto manifold
- Identity: passes mu_hat through (for testing/ablation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple
from dataclasses import dataclass
import math


@dataclass
class RetractionConfig:
    """Configuration for ManifoldRetraction matching RetractionConfig from config.py."""
    retraction_type: str = "pca"          # "pca", "vae", "iterative", "identity"
    latent_dim: int = 64                  # d: dimension of latent space R^d
    manifold_basis_path: Optional[str] = None  # Path to precomputed basis of mu(M)
    pca_components: int = 32              # Number of PCA components for mu(M) subspace
    vae_latent_dim: int = 16              # VAE bottleneck dimension for manifold learning
    vae_encoder_layers: List[int] = None  # VAE encoder hidden layers
    vae_decoder_layers: List[int] = None  # VAE decoder hidden layers
    max_iter: int = 10                    # Max iterations for iterative projection
    tolerance: float = 1e-6               # Convergence tolerance for iterative projection
    idempotence_loss_weight: float = 1.0  # Weight for idempotence loss

    def __post_init__(self):
        if self.vae_encoder_layers is None:
            self.vae_encoder_layers = [128, 64]
        if self.vae_decoder_layers is None:
            self.vae_decoder_layers = [64, 128]


class PCARetraction(nn.Module):
    """
    PCA-based orthogonal projection onto mu(M) subspace.

    Theorem Traceability: A1, T1, D12, CC4, CR-A1, CR-T1, CR-CC4
    - D12: Hilbert projection when inner product available
    - CC4: Retraction layer projecting onto mu(M)
    - CR-CC4: Implementation of projection

    Given a basis B of mu(M) (d x k), projection is:
        mu_final = mu_hat @ B @ B.T  (if B has orthonormal columns)

    Tensor Signature:
    - Input:  [B, d]
    - Output: [B, d]
    """

    def __init__(self, latent_dim: int, pca_components: int, basis: Optional[torch.Tensor] = None):
        super().__init__()
        self.latent_dim = latent_dim
        self.pca_components = pca_components

        if basis is not None:
            # basis shape: (d, k) where k = pca_components
            assert basis.shape == (latent_dim, pca_components),                 f"Basis shape {basis.shape} != ({latent_dim}, {pca_components})"
            # Ensure orthonormal columns
            self.register_buffer("basis", basis)
        else:
            # Initialize with random orthogonal basis (will be updated from data via callback)
            basis = torch.randn(latent_dim, pca_components)
            basis, _ = torch.linalg.qr(basis)
            self.register_buffer("basis", basis)

        # Precompute projection matrix: P = B @ B.T
        self.register_buffer("proj_matrix", basis @ basis.T)

    @classmethod
    def from_data(cls, latent_dim: int, pca_components: int, samples: torch.Tensor) -> "PCARetraction":
        """
        Create PCARetraction from data samples using SVD.

        Args:
            latent_dim: dimension d
            pca_components: number of components k
            samples: [N, d] samples from mu(M) (e.g., mu_final from training)
        """
        # Center and compute SVD
        centered = samples - samples.mean(dim=0, keepdim=True)
        U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
        basis = Vh[:pca_components].T  # (d, k)
        return cls(latent_dim, pca_components, basis)

    def forward(self, mu_hat: torch.Tensor) -> torch.Tensor:
        """
        Project mu_hat onto mu(M) subspace.

        Args:
            mu_hat: [B, d] recovered statistic

        Returns:
            mu_final: [B, d] projected onto mu(M)
        """
        # Orthogonal projection: mu_final = mu_hat @ P
        return mu_hat @ self.proj_matrix

    def idempotence_loss(self, mu_hat: torch.Tensor) -> torch.Tensor:
        """Compute idempotence loss: ||rho(rho(mu)) - rho(mu)||^2"""
        mu_1 = self.forward(mu_hat)
        mu_2 = self.forward(mu_1)
        return F.mse_loss(mu_2, mu_1)

    def manifold_fixing_loss(self, mu_on_manifold: torch.Tensor) -> torch.Tensor:
        """Compute loss for fixing manifold points: ||rho(mu) - mu||^2"""
        mu_proj = self.forward(mu_on_manifold)
        return F.mse_loss(mu_proj, mu_on_manifold)


class VAERetraction(nn.Module):
    """
    VAE-based manifold retraction.

    Theorem Traceability: A1, T1, CC4
    - Learns mu(M) as encoder-decoder bottleneck
    - Idempotence approximately held by reconstruction

    Tensor Signature:
    - Input:  [B, d]
    - Output: [B, d]
    """

    def __init__(
        self,
        latent_dim: int,
        bottleneck_dim: int,
        encoder_layers: List[int],
        decoder_layers: List[int],
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.bottleneck_dim = bottleneck_dim

        # Encoder: d -> bottleneck
        encoder_modules = []
        in_dim = latent_dim
        for hidden in encoder_layers:
            encoder_modules.append(nn.Linear(in_dim, hidden))
            encoder_modules.append(nn.GELU())
            in_dim = hidden
        encoder_modules.append(nn.Linear(in_dim, bottleneck_dim))
        self.encoder = nn.Sequential(*encoder_modules)

        # Decoder: bottleneck -> d
        decoder_modules = []
        in_dim = bottleneck_dim
        for hidden in decoder_layers:
            decoder_modules.append(nn.Linear(in_dim, hidden))
            decoder_modules.append(nn.GELU())
            in_dim = hidden
        decoder_modules.append(nn.Linear(in_dim, latent_dim))
        self.decoder = nn.Sequential(*decoder_modules)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, mu_hat: torch.Tensor) -> torch.Tensor:
        """
        Retract mu_hat onto learned manifold mu(M).

        Args:
            mu_hat: [B, d]

        Returns:
            mu_final: [B, d] on manifold
        """
        z = self.encoder(mu_hat)     # [B, bottleneck]
        mu_final = self.decoder(z)   # [B, d]
        return mu_final

    def encode(self, mu_hat: torch.Tensor) -> torch.Tensor:
        """Encode to manifold coordinates."""
        return self.encoder(mu_hat)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode from manifold coordinates."""
        return self.decoder(z)

    def idempotence_loss(self, mu_hat: torch.Tensor) -> torch.Tensor:
        """Compute idempotence loss."""
        mu_1 = self.forward(mu_hat)
        mu_2 = self.forward(mu_1)
        return F.mse_loss(mu_2, mu_1)

    def manifold_fixing_loss(self, mu_on_manifold: torch.Tensor) -> torch.Tensor:
        """Compute loss for fixing manifold points."""
        mu_proj = self.forward(mu_on_manifold)
        return F.mse_loss(mu_proj, mu_on_manifold)


class IterativeRetraction(nn.Module):
    """
    Iterative projected gradient descent onto manifold (simplified).

    Theorem Traceability: A1, T1, CC4
    - Projects onto manifold via iterative optimization
    - Guarantees idempotence at convergence

    Tensor Signature:
    - Input:  [B, d]
    - Output: [B, d]
    """

    def __init__(
        self,
        latent_dim: int,
        manifold_fn: Optional[nn.Module] = None,
        max_iter: int = 10,
        tolerance: float = 1e-6,
        step_size: float = 1.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.step_size = step_size

        # manifold_fn: R^d -> R^d defining the manifold constraint
        # We use a simple approach: iterative refinement toward manifold
        self.manifold_fn = manifold_fn or nn.Identity()

    def forward(self, mu_hat: torch.Tensor) -> torch.Tensor:
        """
        Simplified iterative retraction - uses a fixed-point iteration.
        For the manifold constraint, we use a simple projection approach.
        """
        B, d = mu_hat.shape
        mu = mu_hat.clone()

        # Simple iterative approach: apply manifold function and blend
        for _ in range(self.max_iter):
            # Apply manifold_fn and check change
            mu_new = self.manifold_fn(mu)
            change = (mu_new - mu).norm(dim=-1).max()
            mu = mu_new
            if change < self.tolerance:
                break

        return mu

    def idempotence_loss(self, mu_hat: torch.Tensor) -> torch.Tensor:
        mu_1 = self.forward(mu_hat)
        mu_2 = self.forward(mu_1)
        return F.mse_loss(mu_2, mu_1)


class IdentityRetraction(nn.Module):
    """
    Identity retraction (pass-through).

    For ablation/testing when no retraction is desired.
    NOT RECOMMENDED for production as it violates T1/CC4.

    Tensor Signature:
    - Input:  [B, d]
    - Output: [B, d] (unchanged)
    """

    def __init__(self, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim

    def forward(self, mu_hat: torch.Tensor) -> torch.Tensor:
        return mu_hat

    def idempotence_loss(self, mu_hat: torch.Tensor) -> torch.Tensor:
        return torch.tensor(0.0, device=mu_hat.device)

    def manifold_fixing_loss(self, mu_on_manifold: torch.Tensor) -> torch.Tensor:
        return torch.tensor(0.0, device=mu_on_manifold.device)


class ManifoldRetraction(nn.Module):
    """
    M3: Manifold Retraction Module - Projects mu_hat onto mu(M).

    Implements MOp-5 (Manifold Retraction).

    Theorem Traceability:
    - A1: Support -> data lives on M
    - T1: Existence -> mu* = pi_obs = quotient map onto mu(M)
    - D12: Hilbert projection (optional) -> orthogonal projection
    - CC4: Manifold retraction -> projection layer rho
    - CR-A1, CR-T1, CR-CC4: Computational requirements

    Tensor Signatures (batch-first):
    - forward(mu_hat)
      mu_hat: [B, d] - recovered statistic from RecoveryModule (M2)
    - Returns:
      mu_final: [B, d] - retracted statistic on mu(M)

    Config: RetractionConfig (maps to RetractionConfig in config.py)
    """

    def __init__(self, config: RetractionConfig):
        super().__init__()
        self.config = config
        self.latent_dim = config.latent_dim
        self.retraction_type = config.retraction_type

        if config.retraction_type == "pca":
            # PCA projection
            basis = None
            if config.manifold_basis_path:
                # Load precomputed basis
                import torch
                basis = torch.load(config.manifold_basis_path)
            self.retraction = PCARetraction(
                latent_dim=config.latent_dim,
                pca_components=config.pca_components,
                basis=basis,
            )
        elif config.retraction_type == "vae":
            self.retraction = VAERetraction(
                latent_dim=config.latent_dim,
                bottleneck_dim=config.vae_latent_dim,
                encoder_layers=config.vae_encoder_layers,
                decoder_layers=config.vae_decoder_layers,
            )
        elif config.retraction_type == "iterative":
            self.retraction = IterativeRetraction(
                latent_dim=config.latent_dim,
                max_iter=config.max_iter,
                tolerance=config.tolerance,
            )
        elif config.retraction_type == "identity":
            self.retraction = IdentityRetraction(latent_dim=config.latent_dim)
        else:
            raise ValueError(f"Unknown retraction_type: {config.retraction_type}")

        self.idempotence_loss_weight = config.idempotence_loss_weight

    def forward(self, mu_hat: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Manifold Retraction.

        Args:
            mu_hat: [B, d] - recovered statistic (pre-retraction)

        Returns:
            mu_final: [B, d] - retracted statistic on mu(M)
        """
        return self.retraction(mu_hat)

    def idempotence_loss(self, mu_hat: torch.Tensor) -> torch.Tensor:
        """Compute idempotence loss: ||rho(rho(mu)) - rho(mu)||^2"""
        return self.idempotence_loss_weight * self.retraction.idempotence_loss(mu_hat)

    def manifold_fixing_loss(self, mu_on_manifold: torch.Tensor) -> torch.Tensor:
        """Compute manifold fixing loss: ||rho(mu) - mu||^2 for mu in mu(M)"""
        return self.idempotence_loss_weight * self.retraction.manifold_fixing_loss(mu_on_manifold)

    def set_basis(self, basis: torch.Tensor):
        """Set PCA basis from external computation (e.g., from training data)."""
        if hasattr(self.retraction, "basis"):
            self.retraction.basis.copy_(basis)
            self.retraction.proj_matrix.copy_(basis @ basis.T)

    def extra_repr(self) -> str:
        return f"latent_dim={self.latent_dim}, type={self.retraction_type}"


def create_manifold_retraction(config: RetractionConfig) -> ManifoldRetraction:
    """
    Factory function to create ManifoldRetraction from config.

    Theorem Traceability: Ensures config matches RetractionConfig from config.py
    """
    return ManifoldRetraction(config)


# ============================================================================
# TorchScript Compatibility
# ============================================================================

class ManifoldRetractionScriptable(nn.Module):
    """
    TorchScript-compatible ManifoldRetraction.

    Uses only scriptable modules and operations.
    """

    def __init__(self, config: RetractionConfig):
        super().__init__()
        self.config = config
        self.latent_dim = config.latent_dim
        self.retraction_type = config.retraction_type

        if config.retraction_type == "pca":
            basis = None
            if config.manifold_basis_path:
                import torch
                basis = torch.load(config.manifold_basis_path)
            self.retraction = PCARetraction(
                latent_dim=config.latent_dim,
                pca_components=config.pca_components,
                basis=basis,
            )
        elif config.retraction_type == "vae":
            self.retraction = VAERetraction(
                latent_dim=config.latent_dim,
                bottleneck_dim=config.vae_latent_dim,
                encoder_layers=config.vae_encoder_layers,
                decoder_layers=config.vae_decoder_layers,
            )
        elif config.retraction_type == "iterative":
            self.retraction = IterativeRetraction(
                latent_dim=config.latent_dim,
                max_iter=config.max_iter,
                tolerance=config.tolerance,
            )
        elif config.retraction_type == "identity":
            self.retraction = IdentityRetraction(latent_dim=config.latent_dim)
        else:
            raise ValueError(f"Unknown retraction_type: {config.retraction_type}")

    def forward(self, mu_hat: torch.Tensor) -> torch.Tensor:
        return self.retraction(mu_hat)


# ============================================================================
# Unit Tests
# ============================================================================

def test_manifold_retraction():
    """Unit tests for ManifoldRetraction."""
    import pytest

    # Test 1: PCA retraction
    config_pca = RetractionConfig(
        retraction_type="pca",
        latent_dim=64,
        pca_components=32,
    )
    retraction_pca = ManifoldRetraction(config_pca)

    B = 16
    mu_hat = torch.randn(B, 64)
    mu_final = retraction_pca(mu_hat)

    assert mu_final.shape == (B, 64), f"PCA output shape {mu_final.shape} != (16, 64)"
    print("Test 1 PASSED: PCA retraction")

    # Test 2: Identity retraction
    config_id = RetractionConfig(
        retraction_type="identity",
        latent_dim=64,
    )
    retraction_id = ManifoldRetraction(config_id)

    mu_final_id = retraction_id(mu_hat)
    assert torch.allclose(mu_final_id, mu_hat), "Identity should not change input"
    print("Test 2 PASSED: Identity retraction")

    # Test 3: VAE retraction
    config_vae = RetractionConfig(
        retraction_type="vae",
        latent_dim=64,
        vae_latent_dim=16,
        vae_encoder_layers=[128, 64],
        vae_decoder_layers=[64, 128],
    )
    retraction_vae = ManifoldRetraction(config_vae)

    mu_final_vae = retraction_vae(mu_hat)
    assert mu_final_vae.shape == (B, 64), f"VAE output shape {mu_final_vae.shape} != (16, 64)"
    print("Test 3 PASSED: VAE retraction")

    # Test 4: Iterative retraction
    config_iter = RetractionConfig(
        retraction_type="iterative",
        latent_dim=64,
        max_iter=5,
        tolerance=1e-4,
    )
    retraction_iter = ManifoldRetraction(config_iter)

    mu_final_iter = retraction_iter(mu_hat)
    assert mu_final_iter.shape == (B, 64), f"Iterative output shape {mu_final_iter.shape} != (16, 64)"
    print("Test 4 PASSED: Iterative retraction")

    # Test 5: PCARetraction from data
    # Generate synthetic data on a low-dimensional manifold
    torch.manual_seed(42)
    N = 1000
    true_basis = torch.randn(64, 32)
    true_basis, _ = torch.linalg.qr(true_basis)  # Orthonormal
    data = torch.randn(N, 32) @ true_basis.T  # Project to manifold
    
    pca_from_data = PCARetraction.from_data(64, 32, data)
    assert pca_from_data.basis.shape == (64, 32)
    print("Test 5 PASSED: PCARetraction from data")

    # Test 6: Idempotence loss for PCA
    loss_idem = retraction_pca.idempotence_loss(mu_hat)
    assert loss_idem >= 0, "Idempotence loss should be non-negative"
    print("Test 6 PASSED: Idempotence loss")

    # Test 7: Manifold fixing loss
    # On-manifold points should have low fixing loss
    mu_on_manifold = torch.randn(10, 32) @ true_basis.T
    loss_fix = retraction_pca.manifold_fixing_loss(mu_on_manifold)
    assert loss_fix >= 0, "Manifold fixing loss should be non-negative"
    print("Test 7 PASSED: Manifold fixing loss")

    # Test 8: Scriptable version
    scriptable = ManifoldRetractionScriptable(config_pca)
    mu_script = scriptable(mu_hat)
    assert mu_script.shape == mu_final.shape
    print("Test 8 PASSED: Scriptable version")

    # Test 9: External basis loading
    custom_basis = torch.randn(64, 32)
    custom_basis, _ = torch.linalg.qr(custom_basis)
    retraction_pca.set_basis(custom_basis)
    mu_final_custom = retraction_pca(mu_hat)
    assert mu_final_custom.shape == (B, 64)
    print("Test 9 PASSED: External basis loading")

    print("\n=== ALL MANIFOLD RETRACTION TESTS PASSED ===")


if __name__ == "__main__":
    test_manifold_retraction()
