"""
HWIN-Net: Hierarchical Water Intelligence Network
==================================================

A theory-derived neural architecture for spatiotemporal water quality prediction
across heterogeneous monitoring platforms.

Main Components:
- M1: Schema Encoder - Platform-specific observation encoding with masking
- M2: Recovery Module - Equivariant recovery maps with weight tying
- M3: Manifold Retraction - Projection onto physically admissible manifold
- M4: Query Head - Prediction with aleatoric/epistemic uncertainty
- M5: Identifiability Gate - Hard refusal for unidentifiable schemas
- M6: No-Leakage Regularizer - Adversarial platform invariance
"""

from hwin_net.utils.config import Config, load_config, validate_config, get_default_config
from hwin_net.utils.seed import set_seed, set_deterministic

__version__ = "0.1.0"
__author__ = "HWIN Team"
__all__ = [
    "Config",
    "load_config",
    "validate_config",
    "get_default_config",
    "set_seed",
    "set_deterministic",
]

# Submodule imports for convenience
from hwin_net import models
from hwin_net import datasets
from hwin_net import losses
from hwin_net import training
from hwin_net import inference
from hwin_net import utils
