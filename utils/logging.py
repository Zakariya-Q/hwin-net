"""
HWIN-Net: Structured Logging

Mathematical Purpose
--------------------
Provides structured logging for training metrics, losses, and hyperparameters
with support for multiple backends (TensorBoard, Weights & Biases, JSONL).

Theory Traceability
-------------------
- Training section of HWIN_Net_Spec.md (logging, checkpointing)
- Reproducibility requirements
"""

import os
import json
import time
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path
import threading
from contextlib import contextmanager


@dataclass
class LoggingConfig:
    log_dir: str = "./logs"
    experiment_name: str = "hwin_net"
    backends: List[str] = field(default_factory=lambda: ["jsonl", "tensorboard"])
    log_level: str = "info"
    flush_every_n_steps: int = 100
    log_histograms: bool = True
    log_gradients: bool = False


class StructuredLogger:
    """
    Structured logger with multiple backends.
    
    Supports:
    - JSONL: Line-delimited JSON for offline analysis
    - TensorBoard: For visualization
    - Weights & Biases: For experiment tracking
    - Console: For real-time monitoring
    """
    
    def __init__(self, config: LoggingConfig):
        self.config = config
        self.log_dir = Path(config.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiment_name = config.experiment_name
        self.flush_every = config.flush_every_n_steps
        self.step_counter = 0
        self.epoch_counter = 0
        
        self._jsonl_file = None
        self._tb_writer = None
        self._wandb_run = None
        self._lock = threading.Lock()
        
        self._init_backends()
    
    def _init_backends(self):
        """Initialize logging backends."""
        # JSONL
        if "jsonl" in self.config.backends:
            jsonl_path = self.log_dir / f"{self.experiment_name}.jsonl"
            self._jsonl_file = open(jsonl_path, 'a', buffering=1)
        
        # TensorBoard
        if "tensorboard" in self.config.backends:
            try:
                from torch.utils.tensorboard import SummaryWriter
                tb_path = self.log_dir / "tensorboard" / self.experiment_name
                self._tb_writer = SummaryWriter(str(tb_path))
            except ImportError:
                print("TensorBoard not available, skipping.")
        
        # Weights & Biases
        if "wandb" in self.config.backends:
            try:
                import wandb
                self._wandb_run = wandb.init(
                    project="hwin_net",
                    name=self.experiment_name,
                    config=None,
                )
            except ImportError:
                print("Weights & Biases not available, skipping.")
    
    def log_scalars(self, prefix: str, metrics: Dict[str, float], step: Optional[int] = None):
        """Log scalar metrics."""
        step = step or self.step_counter
        
        with self._lock:
            # JSONL
            if self._jsonl_file:
                record = {
                    'timestamp': time.time(),
                    'step': step,
                    'epoch': self.epoch_counter,
                    'prefix': prefix,
                }
                record.update(metrics)
                self._jsonl_file.write(json.dumps(record) + '\n')
            
            # TensorBoard
            if self._tb_writer:
                for k, v in metrics.items():
                    self._tb_writer.add_scalar(f"{prefix}/{k}", v, step)
            
            # WandB
            if self._wandb_run:
                log_dict = {f"{prefix}/{k}": v for k, v in metrics.items()}
                log_dict['step'] = step
                self._wandb_run.log(log_dict)
    
    def log_epoch(self, epoch: int, train_metrics: Dict[str, float], val_metrics: Dict[str, float]):
        """Log end-of-epoch metrics."""
        self.epoch_counter = epoch
        self.log_scalars('train', train_metrics, step=epoch)
        self.log_scalars('val', val_metrics, step=epoch)
        self.flush()
    
    def log_histogram(self, name: str, values: Union[torch.Tensor, np.ndarray], step: Optional[int] = None):
        """Log histogram."""
        if not self.config.log_histograms:
            return
        step = step or self.step_counter
        
        if self._tb_writer:
            if isinstance(values, torch.Tensor):
                values = values.detach().cpu().numpy()
            self._tb_writer.add_histogram(name, values, step)
    
    def log_config(self, config: Dict):
        """Log experiment configuration."""
        config_path = self.log_dir / f"{self.experiment_name}_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2, default=str)
        
        if self._tb_writer:
            self._tb_writer.add_text('config', json.dumps(config, indent=2))
    
    def log_model_graph(self, model, input_example):
        """Log model graph to TensorBoard."""
        if self._tb_writer:
            self._tb_writer.add_graph(model, input_example)
    
    def flush(self):
        """Flush all buffers."""
        if self._jsonl_file:
            self._jsonl_file.flush()
        if self._tb_writer:
            self._tb_writer.flush()
    
    def close(self):
        """Close all backends."""
        self.flush()
        if self._jsonl_file:
            self._jsonl_file.close()
        if self._tb_writer:
            self._tb_writer.close()
        if self._wandb_run:
            self._wandb_run.finish()
    
    def increment_step(self):
        """Increment step counter."""
        self.step_counter += 1
        if self.step_counter % self.flush_every == 0:
            self.flush()
    
    @contextmanager
    def epoch(self, epoch: int):
        """Context manager for epoch logging."""
        self.epoch_counter = epoch
        try:
            yield self
        finally:
            pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()


def create_logger(config: Optional[LoggingConfig] = None) -> StructuredLogger:
    """Factory function."""
    return StructuredLogger(config or LoggingConfig())
