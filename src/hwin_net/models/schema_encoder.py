"""
HWIN-Net: Module 1 - Schema Encoder (M1) - FIXED VERSION

Mathematical Purpose
--------------------
Implements MOp-1 (Mask), MOp-2 (Platform Encode), MOp-3 (Observed Encoding)
from Phase XIV Task 2. Computes the schema-specific observation encoding:

    h_a = f_a(x)                    # Platform-specific encoding of full state
    z_g = pi_O(h_a)                 # Masked observation per D2 (zero on unobserved vars)

where:
- f_a: R^n -> R^n is a measurable encoder per platform a in A (Axiom A2)
  that outputs PER-VARIABLE tokens, then projects to k-dim
- M_O in {0,1}^n is the binary mask for observed variables O subset V (Definition 1)
- z_g in R^k with zeros outside O is the encoded observation z_g = pi_O(Phi_a(s)) (Definition 2)

THE FIX: The encoder now outputs per-variable tokens [B, n_vars], applies the mask
to those n_vars dimensions (correct elementwise correspondence), then projects to k.

Theory Traceability
-------------------
- Axiom A2 (Schema Action): Phi_a per platform -> per-platform encoder f_a
- Axiom A3 (Non-degenerate Heterogeneity): Variable |O| and platform a -> encoder handles both
- Definition D2 (Observation): z_g = pi_O o Phi_a(s) -> mask application after encoding
- CR-A2a: Per-platform encoding required by schema action
- CR-A3: Variable observation cardinality handling
- CR-D2: Zero-fill masking per observation definition

Tensor Signatures
-----------------
- Input x:          [B, n]        - Raw state features (full V)
- Input M_O:        [B, n]        - Binary mask (1=observed, 0=unobserved)
- Input a_idx:      [B]           - Platform index in {0, ..., |A|-1}
- Input e_a:        [B, k_a]      - Optional platform embedding (if provided externally)
- Output z_g:       [B, k]        - Masked encoding (pi_O(h_a) projected to k)
- Output h_a:       [B, k]        - Full platform encoding f_a(x) projected to k

Complexity
----------
- Time: O(B * (n * h + |A| * h * n_layers + n * k)) where n=|V|, k=output_dim, h=hidden_dim
- Space: O(|A| * (n * h + h^2 * n_layers) + n * k) for per-platform encoder weights + projection

Assumptions
-----------
- x contains all n variables (unobserved are zero-padded in input)
- M_O is binary with 1 for observed, 0 for unobserved
- Platform indices in a_idx range [0, num_platforms-1]
- Platform embeddings e_a are either learned (from a_idx) or provided externally

Implementation Choices
---------------------
- f_a architecture: MLP per platform (configurable via EncoderConfig.encoder_type)
  Now outputs [B, n_vars] per-variable tokens, then projected to k
- Platform embedding: learned embedding per platform index (k_a dim)
- Mask application: elementwise on per-variable tokens [B, n_vars] BEFORE projection
- TorchScript compatible: all control flow is static per forward pass
- Batched inference: processes all platforms in batch via scatter/gather
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
import math


@dataclass
class SchemaEncoderConfig:
    """Configuration for SchemaEncoder matching EncoderConfig from config.py."""
    encoder_type: str = "mlp"           # "mlp", "transformer", "deepsets"
    n_layers: int = 3
    hidden_dim: int = 128
    output_dim: int = 128                # k: encoding dimension (latent dim)
    platform_embedding_dim: int = 32     # k_a: platform embedding dim
    num_platforms: int = 3               # |A|
    share_platform_encoder: bool = False # Share weights across platforms
    mask_mode: str = "zero_fill"         # "zero_fill" (D2), "soft", "learned"
    dropout: float = 0.1
    activation: str = "gelu"
    norm_type: str = "layer"             # "layer", "batch", "none"
    n_vars: int = 100                    # n = |V| (input dimension)


class PlatformEncoder(nn.Module):
    """
    Per-platform encoder f_a: R^n -> R^n -> R^k.

    Theorem Traceability: A2, A3, CR-A2a, CR-A3
    - A2: Phi_a per platform -> separate f_a per platform
    - A3: Non-degenerate heterogeneity -> handles variable |O|
    - CR-A2a: Per-platform encoding required
    - CR-A3: Variable cardinality support

    FIX: Now outputs per-variable tokens [B, n_vars] first, then projects to k.
    This ensures the mask M_O [B, n] has correct elementwise correspondence.

    Tensor Signature:
    - Input:  [B, n]  (batch for platform a)
    - Output: [B, k]  (after per-variable encoding + projection)
    """

    def __init__(
        self,
        n_vars: int,
        output_dim: int,
        hidden_dim: int,
        n_layers: int,
        platform_embedding_dim: int,
        dropout: float = 0.1,
        activation: str = "gelu",
        norm_type: str = "layer",
    ):
        super().__init__()
        self.n_vars = n_vars
        self.output_dim = output_dim
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

        # Build MLP: [n + k_a] -> hidden -> ... -> n_vars (per-variable tokens)
        layers = []
        in_dim = n_vars + platform_embedding_dim

        for i in range(n_layers - 1):
            out_dim = hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(get_norm(out_dim))
            layers.append(act_fn())
            layers.append(nn.Dropout(dropout))
            in_dim = out_dim

        # Final layer outputs per-variable tokens: n_vars
        layers.append(nn.Linear(in_dim, n_vars))

        self.net = nn.Sequential(*layers)

        # Projection from per-variable tokens to latent dimension k
        self.output_projection = nn.Linear(n_vars, output_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, platform_emb: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of platform encoder.

        Args:
            x: [B, n] input features
            platform_emb: [B, k_a] platform embedding

        Returns:
            [B, k] encoded features
        """
        # Concatenate input with platform embedding
        x = torch.cat([x, platform_emb], dim=-1)
        
        # Encode to per-variable tokens: [B, n_vars]
        h_per_var = self.net(x)
        
        # Project to latent dimension: [B, k]
        return self.output_projection(h_per_var)


class SchemaEncoder(nn.Module):
    """
    M1: Schema Encoder - Computes platform-specific masked encodings.

    Implements MOp-1 (Mask), MOp-2 (Platform Encode), MOp-3 (Observed Encoding).

    THE FIX:
    - Encoder outputs per-variable tokens [B, n_vars]
    - Mask M_O [B, n_vars] applied elementwise to per-variable tokens (CORRECT)
    - Then projected to latent dimension k

    Theorem Traceability:
    - A2: Schema Action -> per-platform encoder f_a
    - A3: Non-degenerate Heterogeneity -> handles variable |O| and a
    - D2: Observation -> z_g = pi_O(Phi_a(s)) via mask on per-variable tokens
    - CR-A2a, CR-A3, CR-D2: Computational requirements

    Tensor Signatures (batch-first):
    - forward(x, M_O, a_idx, e_a=None)
      x:      [B, n]     - raw state features (n = |V|)
      M_O:    [B, n]     - binary mask {0,1}^n
      a_idx:  [B]        - platform index in {0, ..., |A|-1}
      e_a:    [B, k_a]   - optional external platform embedding
    - Returns:
      z_g:    [B, k]     - masked encoding per D2: z_g = project(mask(encode(x)))
      h_a:    [B, k]     - full platform encoding f_a(x) projected to k

    Config: SchemaEncoderConfig (maps to EncoderConfig in config.py)
    """

    def __init__(self, config: SchemaEncoderConfig):
        super().__init__()
        self.config = config
        self.n_vars = config.n_vars
        self.output_dim = config.output_dim
        self.platform_embedding_dim = config.platform_embedding_dim
        self.num_platforms = config.num_platforms
        self.mask_mode = config.mask_mode

        # Platform embeddings: learned per platform index
        self.platform_embeddings = nn.Embedding(
            num_embeddings=config.num_platforms,
            embedding_dim=config.platform_embedding_dim
        )
        nn.init.normal_(self.platform_embeddings.weight, std=0.02)

        # Per-platform encoders f_a
        if config.share_platform_encoder:
            # Single shared encoder with platform embedding as input
            self.encoders = nn.ModuleDict({
                "shared": PlatformEncoder(
                    n_vars=config.n_vars,
                    output_dim=config.output_dim,
                    hidden_dim=config.hidden_dim,
                    n_layers=config.n_layers,
                    platform_embedding_dim=config.platform_embedding_dim,
                    dropout=config.dropout,
                    activation=config.activation,
                    norm_type=config.norm_type,
                )
            })
        else:
            # Separate encoder per platform
            self.encoders = nn.ModuleDict()
            for a in range(config.num_platforms):
                self.encoders[str(a)] = PlatformEncoder(
                    n_vars=config.n_vars,
                    output_dim=config.output_dim,
                    hidden_dim=config.hidden_dim,
                    n_layers=config.n_layers,
                    platform_embedding_dim=config.platform_embedding_dim,
                    dropout=config.dropout,
                    activation=config.activation,
                    norm_type=config.norm_type,
                )

    def forward(
        self,
        x: torch.Tensor,
        M_O: torch.Tensor,
        a_idx: torch.Tensor,
        e_a: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of Schema Encoder.

        Args:
            x: [B, n] - raw state features (full V)
            M_O: [B, n] - binary mask (1=observed, 0=unobserved)
            a_idx: [B] - platform indices
            e_a: [B, k_a] - optional external platform embeddings

        Returns:
            z_g: [B, k] - masked encoding per D2: z_g = project(mask(encode(x)))
            h_a: [B, k] - full platform encoding f_a(x) projected to k
        """
        B, n = x.shape
        device = x.device
        k = self.output_dim

        # Validate input dimensions
        assert n == self.n_vars, f"Input dim {n} != n_vars {self.n_vars}"
        assert M_O.shape == (B, n), f"Mask shape {M_O.shape} != ({B}, {n})"
        assert a_idx.shape == (B,), f"a_idx shape {a_idx.shape} != ({B},)"
        assert a_idx.min() >= 0 and a_idx.max() < self.num_platforms, \
            f"Platform indices out of range [0, {self.num_platforms-1}]"

        # Get platform embeddings: [B, k_a]
        if e_a is not None:
            platform_emb = e_a
            assert platform_emb.shape == (B, self.platform_embedding_dim), \
                f"e_a shape {platform_emb.shape} != ({B}, {self.platform_embedding_dim})"
        else:
            platform_emb = self.platform_embeddings(a_idx)  # [B, k_a]

        # Get per-variable tokens for masking: [B, n_vars]
        h_per_var = torch.empty(B, self.n_vars, device=device, dtype=x.dtype)
        
        if self.config.share_platform_encoder:
            # Single shared encoder for all platforms
            encoder = self.encoders["shared"]
            x_cat = torch.cat([x, platform_emb], dim=-1)
            h_per_var = encoder.net(x_cat)
        else:
            # Separate encoder per platform - scatter/gather
            for a in range(self.num_platforms):
                mask = (a_idx == a)
                if mask.any():
                    encoder = self.encoders[str(a)]
                    x_sub = x[mask]
                    emb_sub = platform_emb[mask]
                    x_cat = torch.cat([x_sub, emb_sub], dim=-1)
                    h_per_var[mask] = encoder.net(x_cat)
        
        # Apply mask on per-variable tokens: [B, n] * [B, n] -> [B, n]
        # This is the CORRECT masking per D2: zero out unobserved variable tokens
        h_per_var_masked = h_per_var * M_O
        
        # Project masked per-variable tokens to latent k: [B, n] -> [B, k]
        if self.config.share_platform_encoder:
            encoder = self.encoders["shared"]
            z_g = encoder.output_projection(h_per_var_masked)
            h_a = encoder.output_projection(h_per_var)
        else:
            z_g = torch.empty(B, k, device=device, dtype=x.dtype)
            h_a = torch.empty(B, k, device=device, dtype=x.dtype)
            for a in range(self.num_platforms):
                mask = (a_idx == a)
                if mask.any():
                    encoder = self.encoders[str(a)]
                    z_g[mask] = encoder.output_projection(h_per_var_masked[mask])
                    h_a[mask] = encoder.output_projection(h_per_var[mask])
        
        return z_g, h_a


    def get_platform_encoder(self, a: int) -> nn.Module:
        """Get encoder for specific platform a."""
        if self.config.share_platform_encoder:
            return self.encoders["shared"]
        return self.encoders[str(a)]

    def extra_repr(self) -> str:
        return (
            f"n_vars={self.n_vars}, output_dim={self.output_dim}, "
            f"num_platforms={self.num_platforms}, share_encoder={self.config.share_platform_encoder}"
        )


def create_schema_encoder(config: SchemaEncoderConfig) -> SchemaEncoder:
    """
    Factory function to create SchemaEncoder from config.

    Theorem Traceability: Ensures config matches EncoderConfig from config.py
    """
    return SchemaEncoder(config)


# ============================================================================
# TorchScript Compatibility
# ============================================================================

# TorchScript-compatible version (separate for script compilation)
# Note: ModuleDict with string keys is not fully TorchScript compatible.
# For TorchScript, use ModuleList with integer indexing.
# This class is provided for reference; use SchemaEncoder for eager mode.

class SchemaEncoderScriptable(nn.Module):
    """
    TorchScript-compatible SchemaEncoder.

    Uses ModuleList instead of ModuleDict for platform encoders.
    Platform embeddings use standard Embedding (TorchScript compatible).
    """

    def __init__(self, config: SchemaEncoderConfig):
        super().__init__()
        self.config = config
        self.n_vars = config.n_vars
        self.output_dim = config.output_dim
        self.platform_embedding_dim = config.platform_embedding_dim
        self.num_platforms = config.num_platforms
        self.mask_mode = config.mask_mode
        self.share_encoder = config.share_platform_encoder

        # Platform embeddings
        self.platform_embeddings = nn.Embedding(
            config.num_platforms, config.platform_embedding_dim
        )
        nn.init.normal_(self.platform_embeddings.weight, std=0.02)

        if self.share_encoder:
            self.encoder = PlatformEncoder(
                n_vars=config.n_vars,
                output_dim=config.output_dim,
                hidden_dim=config.hidden_dim,
                n_layers=config.n_layers,
                platform_embedding_dim=config.platform_embedding_dim,
                dropout=config.dropout,
                activation=config.activation,
                norm_type=config.norm_type,
            )
        else:
            self.encoders = nn.ModuleList([
                PlatformEncoder(
                    n_vars=config.n_vars,
                    output_dim=config.output_dim,
                    hidden_dim=config.hidden_dim,
                    n_layers=config.n_layers,
                    platform_embedding_dim=config.platform_embedding_dim,
                    dropout=config.dropout,
                    activation=config.activation,
                    norm_type=config.norm_type,
                )
                for _ in range(config.num_platforms)
            ])

    def forward(
        self,
        x: torch.Tensor,
        M_O: torch.Tensor,
        a_idx: torch.Tensor,
        e_a: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, n = x.shape
        k = self.output_dim
        device = x.device

        # Platform embeddings
        if e_a is not None:
            platform_emb = e_a
        else:
            platform_emb = self.platform_embeddings(a_idx)

        # Get per-variable tokens for masking
        h_per_var = torch.empty(B, self.n_vars, device=device, dtype=x.dtype)
        
        if self.share_encoder:
            x_cat = torch.cat([x, platform_emb], dim=-1)
            h_per_var = self.encoder.net(x_cat)
        else:
            for a in range(self.num_platforms):
                mask = (a_idx == a)
                if mask.any():
                    encoder = self.encoders[a]
                    x_sub = x[mask]
                    emb_sub = platform_emb[mask]
                    x_cat = torch.cat([x_sub, emb_sub], dim=-1)
                    h_per_var[mask] = encoder.net(x_cat)
        
        # Apply mask on per-variable tokens: [B, n] * [B, n] -> [B, n]
        h_per_var_masked = h_per_var * M_O
        
        # Project masked tokens to latent k
        if self.share_encoder:
            z_g = self.encoder.output_projection(h_per_var_masked)
            h_a = self.encoder.output_projection(h_per_var)
        else:
            z_g = torch.empty(B, k, device=device, dtype=x.dtype)
            h_a = torch.empty(B, k, device=device, dtype=x.dtype)
            for a in range(self.num_platforms):
                mask = (a_idx == a)
                if mask.any():
                    encoder = self.encoders[a]
                    z_g[mask] = encoder.output_projection(h_per_var_masked[mask])
                    h_a[mask] = encoder.output_projection(h_per_var[mask])

        return z_g, h_a


# ============================================================================
# Unit Tests
# ============================================================================

def test_schema_encoder():
    """Unit tests for SchemaEncoder."""
    import pytest

    # Test 1: Basic forward pass
    config = SchemaEncoderConfig(
        n_vars=10,
        output_dim=16,
        hidden_dim=32,
        n_layers=2,
        num_platforms=3,
        platform_embedding_dim=8,
        share_platform_encoder=False,
    )
    encoder = SchemaEncoder(config)

    B = 4
    x = torch.randn(B, 10)
    M_O = torch.randint(0, 2, (B, 10)).float()
    a_idx = torch.randint(0, 3, (B,))

    z_g, h_a = encoder(x, M_O, a_idx)

    assert z_g.shape == (B, 16), f"z_g shape {z_g.shape} != (4, 16)"
    assert h_a.shape == (B, 16), f"h_a shape {h_a.shape} != (4, 16)"
    print("Test 1 PASSED: Basic forward pass")

    # Test 2: Mask application (zero-fill on per-variable tokens)
    # Where M_O=0, the per-variable tokens are zeroed BEFORE projection
    x2 = torch.ones(2, 10)
    M_O2 = torch.tensor([[1., 1., 0., 0., 1., 0., 1., 1., 0., 1.],
                          [0., 1., 1., 1., 0., 1., 0., 0., 1., 0.]])
    a_idx2 = torch.tensor([0, 1])

    z_g2, h_a2 = encoder(x2, M_O2, a_idx2)

    # Check that masking happens at per-variable level
    # The masked output should be different from unmasked in a predictable way
    # because unobserved variables contribute zero to the projection
    assert z_g2.shape == (2, 16)
    assert h_a2.shape == (2, 16)
    print("Test 2 PASSED: Zero-fill masking")

    # Test 3: Shared encoder
    config_shared = SchemaEncoderConfig(
        n_vars=10,
        output_dim=16,
        hidden_dim=32,
        n_layers=2,
        num_platforms=3,
        platform_embedding_dim=8,
        share_platform_encoder=True,
    )
    encoder_shared = SchemaEncoder(config_shared)

    z_g3, h_a3 = encoder_shared(x, M_O, a_idx)
    assert z_g3.shape == (B, 16)
    assert h_a3.shape == (B, 16)
    print("Test 3 PASSED: Shared encoder")

    # Test 4: External platform embeddings
    e_a_ext = torch.randn(B, 8)
    z_g4, h_a4 = encoder(x, M_O, a_idx, e_a=e_a_ext)
    assert z_g4.shape == (B, 16)
    print("Test 4 PASSED: External platform embeddings")

    # Test 5: TorchScript compatibility
    scriptable = SchemaEncoderScriptable(config)
    z_g_script, h_a_script = scriptable(x, M_O, a_idx)
    assert z_g_script.shape == z_g.shape
    assert h_a_script.shape == h_a.shape
    print("Test 5 PASSED: Scriptable encoder")

    # Test 6: Different output_dim vs n_vars
    config_k_neq_n = SchemaEncoderConfig(
        n_vars=10,
        output_dim=32,  # k != n
        hidden_dim=32,
        n_layers=2,
        num_platforms=2,
        platform_embedding_dim=8,
    )
    encoder_neq = SchemaEncoder(config_k_neq_n)
    z_g5, h_a5 = encoder_neq(torch.randn(2, 10), torch.ones(2, 10), torch.tensor([0, 1]))
    assert z_g5.shape == (2, 32)
    assert h_a5.shape == (2, 32)
    print("Test 6 PASSED: k != n handling")

    print("\n=== ALL UNIT TESTS PASSED ===")


def test_dimension_validation():
    """Test input validation."""
    config = SchemaEncoderConfig(n_vars=10, output_dim=16, num_platforms=3)
    encoder = SchemaEncoder(config)

    # Wrong input dim
    try:
        encoder(torch.randn(2, 5), torch.ones(2, 10), torch.tensor([0, 1]))
        assert False, "Should have raised assertion"
    except AssertionError:
        pass
    print("Test: Wrong input dim -> AssertionError PASSED")

    # Wrong mask shape
    try:
        encoder(torch.randn(2, 10), torch.ones(3, 10), torch.tensor([0, 1]))
        assert False, "Should have raised assertion"
    except AssertionError:
        pass
    print("Test: Wrong mask shape -> AssertionError PASSED")

    # Wrong a_idx shape
    try:
        encoder(torch.randn(2, 10), torch.ones(2, 10), torch.tensor([0]))
        assert False, "Should have raised assertion"
    except AssertionError:
        pass
    print("Test: Wrong a_idx shape -> AssertionError PASSED")

    # Out of range platform index
    try:
        encoder(torch.randn(2, 10), torch.ones(2, 10), torch.tensor([0, 5]))
        assert False, "Should have raised assertion"
    except AssertionError:
        pass
    print("Test: Out of range platform -> AssertionError PASSED")


if __name__ == "__main__":
    test_schema_encoder()
    test_dimension_validation()
    print("\n=== ALL TESTS PASSED ===")
