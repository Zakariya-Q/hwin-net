"""
HWIN-Net: Inference Package
============================

Inference and uncertainty quantification:
- InferenceEngine: Main inference pipeline
- MCDropoutUncertainty: MC dropout uncertainty
- EnsembleUncertainty: Ensemble-based uncertainty
"""

from hwin_net.inference.inference import InferenceEngine, InferenceConfig, InferenceOutput, create_inference
from hwin_net.inference.uncertainty import (
    MCDropoutUncertainty, 
    EnsembleUncertainty,
)

__all__ = [
    "InferenceEngine",
    "InferenceConfig",
    "InferenceOutput", 
    "create_inference",
    "MCDropoutUncertainty",
    "EnsembleUncertainty",
]
