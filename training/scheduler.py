"""
HWIN-Net: Learning Rate Schedulers

Mathematical Purpose
--------------------
Implements learning rate schedules for TTUR training:
- Cosine annealing with warmup (main, recovery)
- Constant or step decay (adversarial)
- ReduceLROnPlateau for validation-based scheduling

Theory Traceability
-------------------
- Training section of HWIN_Net_Spec.md
- Standard deep learning practice for stable convergence
"""

import torch
from typing import Optional
from dataclasses import dataclass
import math


@dataclass
class SchedulerConfig:
    scheduler_type: str = "cosine"  # "cosine", "step", "plateau", "constant"
    warmup_epochs: int = 5
    max_epochs: int = 100
    min_lr: float = 1e-6
    # For step scheduler
    step_size: int = 30
    gamma: float = 0.1
    # For plateau scheduler
    plateau_patience: int = 10
    plateau_factor: float = 0.5
    plateau_mode: str = "min"


class WarmupCosineScheduler:
    """
    Cosine annealing with linear warmup.

    lr = warmup_lr * (epoch / warmup_epochs)           during warmup
    lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(pi * (epoch - warmup) / (max_epochs - warmup)))  after
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        base_lr: float,
        config: SchedulerConfig,
    ):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.config = config
        self.current_epoch = 0

    def step(self, epoch: Optional[int] = None):
        if epoch is not None:
            self.current_epoch = epoch

        lr = self._get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.current_epoch += 1

    def _get_lr(self) -> float:
        epoch = self.current_epoch
        warmup = self.config.warmup_epochs
        max_epochs = self.config.max_epochs
        min_lr = self.config.min_lr

        if epoch < warmup:
            # Linear warmup
            return self.base_lr * (epoch + 1) / warmup

        # Cosine annealing
        progress = (epoch - warmup) / max(1, max_epochs - warmup)
        cos_factor = 0.5 * (1 + math.cos(math.pi * progress))
        return min_lr + (self.base_lr - min_lr) * cos_factor

    def state_dict(self) -> dict:
        return {
            'current_epoch': self.current_epoch,
            'base_lr': self.base_lr,
        }

    def load_state_dict(self, state_dict: dict):
        self.current_epoch = state_dict['current_epoch']
        self.base_lr = state_dict['base_lr']


class StepScheduler:
    """StepLR scheduler."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        base_lr: float,
        config: SchedulerConfig,
    ):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.config = config
        self.current_epoch = 0

    def step(self, epoch: Optional[int] = None):
        if epoch is not None:
            self.current_epoch = epoch
        lr = self.base_lr * (self.config.gamma ** (self.current_epoch // self.config.step_size))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        self.current_epoch += 1

    def state_dict(self) -> dict:
        return {'current_epoch': self.current_epoch, 'base_lr': self.base_lr}

    def load_state_dict(self, state_dict: dict):
        self.current_epoch = state_dict['current_epoch']
        self.base_lr = state_dict['base_lr']


class PlateauScheduler:
    """ReduceLROnPlateau wrapper."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        base_lr: float,
        config: SchedulerConfig,
    ):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.config = config
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=config.plateau_mode,
            factor=config.plateau_factor,
            patience=config.plateau_patience,
            min_lr=config.min_lr,
            verbose=False,
        )
        self.current_epoch = 0

    def step(self, metric: float = None, epoch: Optional[int] = None):
        if epoch is not None:
            self.current_epoch = epoch
        if metric is not None:
            self.scheduler.step(metric)
        else:
            # Step without metric (just epoch-based)
            self.current_epoch += 1

    def state_dict(self) -> dict:
        return {
            'current_epoch': self.current_epoch,
            'base_lr': self.base_lr,
            'scheduler': self.scheduler.state_dict(),
        }

    def load_state_dict(self, state_dict: dict):
        self.current_epoch = state_dict['current_epoch']
        self.base_lr = state_dict['base_lr']
        self.scheduler.load_state_dict(state_dict['scheduler'])


class ConstantScheduler:
    """Constant learning rate."""

    def __init__(self, optimizer: torch.optim.Optimizer, base_lr: float):
        self.optimizer = optimizer
        self.base_lr = base_lr
        for pg in optimizer.param_groups:
            pg['lr'] = base_lr

    def step(self, epoch: Optional[int] = None, metric: float = None):
        pass

    def state_dict(self) -> dict:
        return {'base_lr': self.base_lr}

    def load_state_dict(self, state_dict: dict):
        self.base_lr = state_dict['base_lr']


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    base_lr: float,
    config: SchedulerConfig,
):
    """Factory to create scheduler."""
    if config.scheduler_type == "cosine":
        return WarmupCosineScheduler(optimizer, base_lr, config)
    elif config.scheduler_type == "step":
        return StepScheduler(optimizer, base_lr, config)
    elif config.scheduler_type == "plateau":
        return PlateauScheduler(optimizer, base_lr, config)
    elif config.scheduler_type == "constant":
        return ConstantScheduler(optimizer, base_lr)
    else:
        raise ValueError(f"Unknown scheduler_type: {config.scheduler_type}")


class TTURScheduler:
    """
    TTUR-aware scheduler managing three separate schedulers.
    """

    def __init__(
        self,
        main_optimizer: torch.optim.Optimizer,
        recovery_optimizer: torch.optim.Optimizer,
        adversarial_optimizer: torch.optim.Optimizer,
        config: SchedulerConfig,
        lr_main: float,
        lr_recovery: float,
        lr_adversarial: float,
    ):
        self.schedulers = {
            'main': create_scheduler(main_optimizer, lr_main, config),
            'recovery': create_scheduler(recovery_optimizer, lr_recovery, config),
            'adversarial': create_scheduler(adversarial_optimizer, lr_adversarial, config),
        }

    def step(self, epoch: int, val_metric: Optional[float] = None):
        for name, sched in self.schedulers.items():
            if isinstance(sched, PlateauScheduler):
                sched.step(metric=val_metric, epoch=epoch)
            else:
                sched.step(epoch=epoch)

    def state_dict(self) -> dict:
        return {k: v.state_dict() for k, v in self.schedulers.items()}

    def load_state_dict(self, state_dict: dict):
        for k, v in state_dict.items():
            self.schedulers[k].load_state_dict(v)

    def get_lrs(self) -> dict:
        return {k: sched.optimizer.param_groups[0]['lr'] for k, sched in self.schedulers.items()}
