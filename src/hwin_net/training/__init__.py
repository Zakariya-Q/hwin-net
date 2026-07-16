"""
HWIN-Net: Training Package
===========================

Training utilities:
- TTUROptimizer: OneAdam optimizer with layer-wise learning rates
- TTURScheduler: Cosine/Warmup/Plateau schedulers
- Train_HWIN: Main training loop with logging/checkpointing
"""

from hwin_net.training.optimizer import TTUROptimizer, TTUROptimizerConfig, create_ttur_optimizer
from hwin_net.training.scheduler import TTURScheduler, SchedulerConfig, create_scheduler
from hwin_net.training.trainer import Train_HWIN, HWINNetTrainer, TrainerConfig

__all__ = [
    "TTUROptimizer",
    "TTUROptimizerConfig",
    "create_ttur_optimizer",
    "TTURScheduler",
    "SchedulerConfig",
    "create_scheduler",
    "Train_HWIN",
    "HWINNetTrainer",
    "TrainerConfig",
]
