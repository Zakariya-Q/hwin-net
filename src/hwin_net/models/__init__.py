"""
HWIN-Net: Models Package
========================

Core model components implementing the theory-derived architecture:
- M1: SchemaEncoder (platform-specific encoding with masking)
- M2: RecoveryModule (equivariant recovery maps)
- M3: ManifoldRetraction (projection onto physical manifold)
- M4: QueryHead (prediction with uncertainty decomposition)
- M5: IdentifiabilityGate (hard refusal for unidentifiable schemas)
- M6: NoLeakageRegularizer (adversarial platform invariance)
- HWINNet: Full integrated model
"""

from hwin_net.models.hwin_net import HWINNet
from hwin_net.models.schema_encoder import SchemaEncoder, SchemaEncoderConfig
from hwin_net.models.recovery_module import RecoveryModule, RecoveryConfig
from hwin_net.models.manifold_retraction import ManifoldRetraction, RetractionConfig
from hwin_net.models.query_head import QueryHead, QueryHeadConfig
from hwin_net.models.identifiability_gate import IdentifiabilityGate, GateConfig
from hwin_net.models.no_leakage import NoLeakageRegularizer, NoLeakageRegularizerConfig

__all__ = [
    "HWINNet",
    "SchemaEncoder",
    "SchemaEncoderConfig",
    "RecoveryModule",
    "RecoveryConfig",
    "ManifoldRetraction",
    "RetractionConfig",
    "QueryHead",
    "QueryHeadConfig",
    "IdentifiabilityGate",
    "GateConfig",
    "NoLeakageRegularizer",
    "NoLeakageRegularizerConfig",
]
