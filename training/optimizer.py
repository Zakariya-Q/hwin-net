"""
HWIN-Net: TTUR Optimizer Configuration

Mathematical Purpose
--------------------
Implements Two-Time-Scale Update Rule (TTUR) for training HWIN-Net
with adversarial no-leakage regularizer (M6).

Per Training Spec:
- Main params (theta, psi): lr_main, AdamW
- Recovery params (theta_T): lr_recovery, AdamW
- Adversarial disc: lr_adversarial, Adam
- Gradient clipping, weight decay

Theory Traceability
-------------------
- Training section of HWIN_Net_Spec.md
- TTUR for GAN-like adversarial training (M6)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import OrderedDict


@dataclass
class TTUROptimizerConfig:
    use_ttur: bool = True
    lr_main: float = 1e-3          # theta (encoder, query head), psi
    lr_recovery: float = 5e-4      # theta_T (recovery module)
    lr_adversarial: float = 2e-4   # discriminator (no-leakage)
    optimizer_type: str = "adamw"  # "adamw", "adam"
    weight_decay: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    grad_clip_norm: float = 1.0


class TTUROptimizer:
    """
    Two-Time-Scale Update Rule optimizer wrapper.

    Manages three separate optimizers:
    1. main_optimizer: encoder (M1), query head (M4), gate (M5), retraction (M3)
    2. recovery_optimizer: recovery module (M2)
    3. adversarial_optimizer: discriminator (M6)

    Memory-efficient: shares optimizer state structure
    """

    def __init__(
        self,
        config: TTUROptimizerConfig,
        model_params: Dict[str, List[torch.nn.Parameter]],
    ):
        """
        Args:
            config: TTUROptimizerConfig
            model_params: Dict with keys:
                - 'main': list of params for main optimizer
                - 'recovery': list of params for recovery optimizer
                - 'adversarial': list of params for adversarial optimizer
        """
        self.config = config
        self.use_ttur = config.use_ttur

        if config.optimizer_type == "adamw":
            opt_class = torch.optim.AdamW
        elif config.optimizer_type == "adam":
            opt_class = torch.optim.Adam
        else:
            raise ValueError(f"Unknown optimizer_type: {config.optimizer_type}")

        # Main optimizer (encoder, query head, gate, retraction)
        self.main_optimizer = opt_class(
            model_params.get('main', []),
            lr=config.lr_main,
            weight_decay=config.weight_decay,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
        )

        # Recovery optimizer (recovery module)
        self.recovery_optimizer = opt_class(
            model_params.get('recovery', []),
            lr=config.lr_recovery,
            weight_decay=config.weight_decay,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
        )

        # Adversarial optimizer (discriminator)
        self.adversarial_optimizer = opt_class(
            model_params.get('adversarial', []),
            lr=config.lr_adversarial,
            weight_decay=0.0,  # No weight decay for discriminator typically
            betas=(config.beta1, config.beta2),
            eps=config.eps,
        )

        self.grad_clip_norm = config.grad_clip_norm

    def zero_grad(self, set_to_none: bool = True):
        """Zero all optimizer gradients."""
        self.main_optimizer.zero_grad(set_to_none=set_to_none)
        self.recovery_optimizer.zero_grad(set_to_none=set_to_none)
        self.adversarial_optimizer.zero_grad(set_to_none=set_to_none)

    def step_main(self):
        """Step main optimizer with gradient clipping."""
        if self.grad_clip_norm > 0:
            # Get all main parameters
            main_params = []
            for group in self.main_optimizer.param_groups:
                main_params.extend(group['params'])
            torch.nn.utils.clip_grad_norm_(main_params, self.grad_clip_norm)
        self.main_optimizer.step()

    def step_recovery(self):
        """Step recovery optimizer with gradient clipping."""
        if self.grad_clip_norm > 0:
            recovery_params = []
            for group in self.recovery_optimizer.param_groups:
                recovery_params.extend(group['params'])
            torch.nn.utils.clip_grad_norm_(recovery_params, self.grad_clip_norm)
        self.recovery_optimizer.step()

    def step_adversarial(self):
        """Step adversarial optimizer with gradient clipping."""
        if self.grad_clip_norm > 0:
            adv_params = []
            for group in self.adversarial_optimizer.param_groups:
                adv_params.extend(group['params'])
            torch.nn.utils.clip_grad_norm_(adv_params, self.grad_clip_norm)
        self.adversarial_optimizer.step()

    def step_all(self):
        """Step all optimizers (deprecated - use separate steps for TTUR)."""
        self.step_main()
        self.step_recovery()
        self.step_adversarial()

    def state_dict(self) -> Dict:
        """Return state dicts for all optimizers."""
        return {
            'main': self.main_optimizer.state_dict(),
            'recovery': self.recovery_optimizer.state_dict(),
            'adversarial': self.adversarial_optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict: Dict):
        """Load state dicts for all optimizers."""
        self.main_optimizer.load_state_dict(state_dict['main'])
        self.recovery_optimizer.load_state_dict(state_dict['recovery'])
        self.adversarial_optimizer.load_state_dict(state_dict['adversarial'])

    def get_lrs(self) -> Dict[str, float]:
        """Get current learning rates."""
        return {
            'main': self.main_optimizer.param_groups[0]['lr'],
            'recovery': self.recovery_optimizer.param_groups[0]['lr'],
            'adversarial': self.adversarial_optimizer.param_groups[0]['lr'],
        }

    def set_lr(self, lr_main: Optional[float] = None,
               lr_recovery: Optional[float] = None,
               lr_adversarial: Optional[float] = None):
        """Set learning rates."""
        if lr_main is not None:
            for pg in self.main_optimizer.param_groups:
                pg['lr'] = lr_main
        if lr_recovery is not None:
            for pg in self.recovery_optimizer.param_groups:
                pg['lr'] = lr_recovery
        if lr_adversarial is not None:
            for pg in self.adversarial_optimizer.param_groups:
                pg['lr'] = lr_adversarial


def extract_params_for_ttur(model, encoder_modules, recovery_module, no_leakage_module):
    """
    Helper to extract parameter groups for TTUR from HWINNet model.

    Args:
        model: HWINNet instance
        encoder_modules: list of module names for main optimizer
        recovery_module: module name for recovery optimizer
        no_leakage_module: module name for adversarial optimizer

    Returns:
        Dict with 'main', 'recovery', 'adversarial' parameter lists
    """
    main_params = []
    recovery_params = []
    adversarial_params = []

    # Main: schema_encoder, manifold_retraction, query_head, identifiability_gate
    for name in ['schema_encoder', 'manifold_retraction', 'query_head', 'identifiability_gate']:
        if hasattr(model, name):
            main_params.extend(list(getattr(model, name).parameters()))

    # Recovery: recovery_module
    if hasattr(model, 'recovery_module'):
        recovery_params.extend(list(model.recovery_module.parameters()))

    # Adversarial: no_leakage discriminator
    if hasattr(model, 'no_leakage') and hasattr(model.no_leakage, 'discriminator'):
        adversarial_params.extend(list(model.no_leakage.discriminator.parameters()))

    # Handle case when discriminator is disabled (no_leakage.enabled = False)
    if len(adversarial_params) == 0:
        # Add a dummy parameter that requires grad but is detached
        dummy = torch.nn.Parameter(torch.tensor(0.0), requires_grad=True)
        adversarial_params.append(dummy)

    return {
        'main': main_params,
        'recovery': recovery_params,
        'adversarial': adversarial_params,
    }


def create_ttur_optimizer(config: TTUROptimizerConfig, model) -> TTUROptimizer:
    """Factory function to create TTUROptimizer from HWINNet model."""
    param_groups = extract_params_for_ttur(
        model,
        encoder_modules=['schema_encoder'],
        recovery_module='recovery_module',
        no_leakage_module='no_leakage'
    )
    return TTUROptimizer(config, param_groups)
