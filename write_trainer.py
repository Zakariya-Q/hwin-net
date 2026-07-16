import os
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Tuple, Callable
from dataclasses import dataclass, field
from pathlib import Path
import time
from collections import defaultdict

from models.hwin_net import HWINNet
from losses.total_loss import TotalLoss, TotalLossConfig, create_total_loss
from training.optimizer import TTUROptimizer, TTUROptimizerConfig, create_ttur_optimizer
from training.scheduler import TTURScheduler, SchedulerConfig
from utils.config import Config, TrainingConfig
from utils.seed import set_seed, set_deterministic
import logging
from utils.structured_logging import setup_logging, log_metrics

