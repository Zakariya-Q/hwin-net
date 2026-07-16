import os
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Tuple, Callable
from dataclasses import dataclass, field
from pathlib import Path
import time
import json
from collections import defaultdict

from hwin_net.models.hwin_net import HWINNet
from hwin_net.losses.total_loss import TotalLoss, TotalLossConfig, create_total_loss
from hwin_net.training.optimizer import TTUROptimizer, TTUROptimizerConfig, create_ttur_optimizer
from hwin_net.training.scheduler import TTURScheduler, SchedulerConfig
from hwin_net.utils.config import Config, TrainingConfig, validate_config
from hwin_net.utils.seed import set_seed, set_deterministic
import logging
from hwin_net.utils.structured_logging import setup_logging, log_metrics


@dataclass
class TrainState:
    epoch: int = 0
    global_step: int = 0
    best_val_metric: float = float('inf')
    epochs_without_improvement: int = 0
    train_history: List[Dict] = field(default_factory=list)
    val_history: List[Dict] = field(default_factory=list)
    
    def state_dict(self) -> Dict[str, Any]:
        return {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'best_val_metric': self.best_val_metric,
            'epochs_without_improvement': self.epochs_without_improvement,
            'train_history': self.train_history,
            'val_history': self.val_history,
        }
    
    def load_state_dict(self, state: Dict[str, Any]):
        self.epoch = state.get('epoch', 0)
        self.global_step = state.get('global_step', 0)
        self.best_val_metric = state.get('best_val_metric', float('inf'))
        self.epochs_without_improvement = state.get('epochs_without_improvement', 0)
        self.train_history = state.get('train_history', [])
        self.val_history = state.get('val_history', [])


@dataclass
class TrainerConfig:
    max_epochs: int = 100
    batch_size: int = 32
    val_batch_size: int = 64
    validate_every_n_epochs: int = 1
    checkpoint_dir: str = './checkpoints'
    save_every_n_epochs: int = 10
    keep_last_n_checkpoints: int = 3
    resume_from: Optional[str] = None
    early_stopping_patience: int = 20
    early_stopping_metric: str = 'val_loss'
    early_stopping_mode: str = 'min'
    use_mixed_precision: bool = True
    seed: int = 42
    deterministic: bool = True
    device: str = 'auto'
    num_workers: int = 4
    pin_memory: bool = True
    log_level: str = 'info'
    log_dir: str = './logs'
    experiment_name: str = 'hwin_net'
    lambda_pred: float = 1.0
    lambda_rec: float = 1.0
    lambda_noleak: float = 0.1
    lambda_equiv: float = 0.1
    lambda_complex: float = 0.0001
    loss: Any = None


class PCABasisCallback:
    def __init__(
        self,
        manifold_retraction_module,
        pca_components: int = 32,
        update_every_n_epochs: int = 5,
    ):
        self.manifold_retraction = manifold_retraction_module
        self.pca_components = pca_components
        self.update_every_n_epochs = update_every_n_epochs
        self.collected_mu_final = []
        self.logger = logging.getLogger(__name__)
    
    def on_batch_end(self, mu_final: torch.Tensor):
        if len(self.collected_mu_final) < 1000:
            self.collected_mu_final.append(mu_final.detach().cpu())
    
    def update_basis(self):
        if len(self.collected_mu_final) == 0:
            return
        all_mu = torch.cat(self.collected_mu_final, dim=0)
        mean = all_mu.mean(dim=0)
        centered = all_mu - mean
        # PCA may return fewer components than requested if samples < components
        actual_q = min(self.pca_components, centered.shape[0], centered.shape[1])
        U, S, V = torch.pca_lowrank(centered, q=actual_q)
        # V has shape (d, q) = (latent_dim, actual_q) - this is the correct basis
        # Only update if the module supports set_basis
        if hasattr(self.manifold_retraction, 'set_basis'):
            self.manifold_retraction.set_basis(V)
            self.logger.info('Updated PCA basis: ' + str(U.shape) + ', components=' + str(self.pca_components))
        self.collected_mu_final = []


class HWINNetTrainer:
    def __init__(
        self,
        model: HWINNet,
        config,
        train_loader,
        val_loader,
        logger: Optional[logging.Logger] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.logger = logger or logging.getLogger(__name__)
        
        if hasattr(config, 'training'):
            self.training_config = config.training
            self.lambda_pred = config.loss.lambda_pred
            self.lambda_rec = config.loss.lambda_rec
            self.lambda_noleak = config.loss.lambda_noleak
            self.lambda_equiv = config.loss.lambda_equiv
            self.lambda_complex = config.loss.lambda_complex
            self.loss_config = config.loss
            self.encoder_config = config.encoder
            self.recovery_config = config.recovery
        else:
            self.training_config = config
            self.lambda_pred = config.lambda_pred
            self.lambda_rec = config.lambda_rec
            self.lambda_noleak = config.lambda_noleak
            self.lambda_equiv = config.lambda_equiv
            self.lambda_complex = config.lambda_complex
            self.loss_config = config.loss
            if hasattr(config, 'encoder'):
                self.encoder_config = config.encoder
            else:
                raise ValueError('Config must have encoder attribute for loss creation')
            if hasattr(config, 'recovery'):
                self.recovery_config = config.recovery
            else:
                raise ValueError('Config must have recovery attribute for loss creation')

        self.checkpoint_dir = Path(self.training_config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.equivariance_warmup_epochs = getattr(
            self.training_config, 'equivariance_warmup_epochs', 10
        )
        
        self.device = self._resolve_device()
        self.model = self.model.to(self.device)
        
        if self.training_config.deterministic:
            set_deterministic(True)
        set_seed(self.training_config.seed)
        
        self.loss_fn = create_total_loss(
            TotalLossConfig(
                lambda_pred=self.lambda_pred,
                lambda_rec=self.lambda_rec,
                lambda_noleak=self.lambda_noleak,
                lambda_equiv=self.lambda_equiv,
                lambda_complex=self.lambda_complex,
                pred_loss_type=self.loss_config.pred_loss_type,
                rec_loss_type=self.loss_config.rec_loss_type,
                equiv_loss_type=self.loss_config.equiv_loss_type,
            ),
            z_dim=self.encoder_config.output_dim,
            num_platforms=self.encoder_config.num_platforms,
            latent_dim=self.recovery_config.latent_dim,
        ).to(self.device)
        
        self.optimizer = create_ttur_optimizer(
            TTUROptimizerConfig(
                use_ttur=self.training_config.use_ttur,
                lr_main=self.training_config.lr_main,
                lr_recovery=self.training_config.lr_recovery,
                lr_adversarial=self.training_config.lr_adversarial,
                optimizer_type=self.training_config.optimizer_type,
                weight_decay=self.training_config.weight_decay,
                beta1=self.training_config.beta1,
                beta2=self.training_config.beta2,
                eps=self.training_config.eps,
                grad_clip_norm=self.training_config.grad_clip_norm,
            ),
            model
        )
        
        self.scheduler = TTURScheduler(
            main_optimizer=self.optimizer.main_optimizer,
            recovery_optimizer=self.optimizer.recovery_optimizer,
            adversarial_optimizer=self.optimizer.adversarial_optimizer,
            config=SchedulerConfig(
                scheduler_type=self.training_config.scheduler_type,
                warmup_epochs=self.training_config.warmup_epochs,
                max_epochs=self.training_config.max_epochs,
                min_lr=self.training_config.min_lr,
                step_size=self.training_config.step_size,
                gamma=self.training_config.gamma,
                plateau_patience=self.training_config.plateau_patience,
                plateau_factor=self.training_config.plateau_factor,
                plateau_mode=getattr(self.training_config, "plateau_mode", "min"),
            ),
            lr_main=self.training_config.lr_main,
            lr_recovery=self.training_config.lr_recovery,
            lr_adversarial=self.training_config.lr_adversarial,
        )
        
        self.use_amp = (
            self.training_config.use_mixed_precision and self.device.type == 'cuda'
        )
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        
        self.state = TrainState()
        
        self.pca_callback = PCABasisCallback(
           self.model.manifold_retraction,
            pca_components=getattr(config, 'retraction', None) and config.retraction.pca_components or 32,
           update_every_n_epochs=5,
       )
        
        if self.training_config.resume_from:
            self.load_checkpoint(self.training_config.resume_from)

    def _resolve_device(self) -> torch.device:
        if self.training_config.device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(self.training_config.device)
    
    def _compute_equivariance_factor(self, epoch: int) -> float:
        if epoch < self.equivariance_warmup_epochs:
            return epoch / max(1, self.equivariance_warmup_epochs)
        return 1.0
    
    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        epoch_losses = defaultdict(float)
        num_batches = 0
        
        equiv_factor = self._compute_equivariance_factor(self.state.epoch)
        
        for batch in self.train_loader:
            x, M_O, a_idx, y = self._prepare_batch(batch)
            
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                outputs = self.model.forward(
                    x=x,
                    M_O=M_O,
                    a_idx=a_idx,
                    y=y,
                    training=True,
                    equivariance_warmup_factor=equiv_factor,
                )
                
                loss_dict = self.loss_fn(
                    q_out=outputs['q_out'],
                    q_hat=outputs['q_hat'],
                    sigma2_aleat=outputs.get('sigma2_aleat'),
                    sigma2_total=outputs.get('sigma2_total'),
                    y=y,
                    mu_hat=outputs['mu_hat'],
                    mu_final=outputs['mu_final'],
                    z_g=outputs['z_g'],
                    a_idx=a_idx,
                    training=True,
                )
                main_loss = loss_dict['total_loss']
                
                self.optimizer.zero_grad()
                disc_out = self.loss_fn.noleak_loss.discriminator_step(outputs['z_g'], a_idx)
                disc_loss = disc_out['disc_loss']
                disc_acc = disc_out.get('disc_acc', 0.0)
                if self.use_amp:
                    self.scaler.scale(disc_loss).backward()
                else:
                    disc_loss.backward()
                self.optimizer.step_adversarial()
                
                self.optimizer.zero_grad()
                _ = self.loss_fn.noleak_loss.encoder_step(outputs['z_g'], a_idx)
                noleak_loss = loss_dict['noleak_loss']
                
                total_loss = main_loss + noleak_loss
                
                if self.use_amp:
                    self.scaler.scale(total_loss).backward()
                else:
                    total_loss.backward()
                
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer.main_optimizer)
                    self.scaler.unscale_(self.optimizer.recovery_optimizer)
                    self.optimizer.step_main()
                    self.optimizer.step_recovery()
                    self.scaler.update()
                else:
                    self.optimizer.step_main()
                    self.optimizer.step_recovery()
            
            self.pca_callback.on_batch_end(outputs['mu_final'])
            
            epoch_losses['total_loss'] += total_loss.item()
            for k, v in loss_dict.items():
                if isinstance(v, torch.Tensor):
                    epoch_losses[k] += v.item()
                else:
                    epoch_losses[k] += v
            epoch_losses['disc_loss'] += disc_loss.item() if isinstance(disc_loss, torch.Tensor) else disc_loss
            epoch_losses['disc_acc'] += disc_acc.item() if isinstance(disc_acc, torch.Tensor) else disc_acc
            epoch_losses['equivariance_factor'] = equiv_factor
            num_batches += 1
            self.state.global_step += 1
        
        avg_losses = {k: v / max(1, num_batches) for k, v in epoch_losses.items()}
        return avg_losses

    def _prepare_batch(self, batch) -> Tuple:
        if isinstance(batch, (list, tuple)):
            if len(batch) == 4:
                x, M_O, a_idx, y = batch
            elif len(batch) == 3:
                x, M_O, a_idx = batch
                y = None
            else:
                raise ValueError('Unexpected batch format: ' + str(len(batch)) + ' elements')
        elif isinstance(batch, dict):
            x = batch['x']
            M_O = batch['M_O']
            a_idx = batch['a_idx']
            y = batch.get('y')
        else:
            raise ValueError('Unexpected batch type: ' + str(type(batch)))
        
        x = x.to(self.device, non_blocking=True)
        M_O = M_O.to(self.device, non_blocking=True)
        a_idx = a_idx.to(self.device, non_blocking=True)
        if y is not None:
            y = y.to(self.device, non_blocking=True)
        
        return x, M_O, a_idx, y
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        self.model.eval()
        epoch_losses = defaultdict(float)
        num_batches = 0
        
        for batch in self.val_loader:
            x, M_O, a_idx, y = self._prepare_batch(batch)
            
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                outputs = self.model.forward(
                    x=x,
                    M_O=M_O,
                    a_idx=a_idx,
                    y=y,
                    training=False,
                    equivariance_warmup_factor=1.0,
                )
                
                loss_dict = self.loss_fn(
                    q_out=outputs['q_out'],
                    q_hat=outputs['q_hat'],
                    sigma2_aleat=outputs.get('sigma2_aleat'),
                    sigma2_total=outputs.get('sigma2_total'),
                    y=y,
                    mu_hat=outputs['mu_hat'],
                    mu_final=outputs['mu_final'],
                    z_g=outputs['z_g'],
                    a_idx=a_idx,
                    training=False,
                )
            
            for k, v in loss_dict.items():
                if isinstance(v, torch.Tensor):
                    epoch_losses[k] += v.item()
                else:
                    epoch_losses[k] += v
            num_batches += 1
        
        avg_losses = {k: v / max(1, num_batches) for k, v in epoch_losses.items()}
        return avg_losses
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        checkpoint = {
            'epoch': epoch,
            'global_step': self.state.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_state': self.state.state_dict(),
            'config': {
                'model_config': self.model.config,
                'trainer_config': self.training_config,
            },
        }
        
        if self.use_amp:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        if epoch % self.training_config.save_every_n_epochs == 0:
            path = self.checkpoint_dir / ('checkpoint_epoch_' + str(epoch) + '.pt')
            torch.save(checkpoint, path)
            self._cleanup_checkpoints()
        
        if is_best:
            path = self.checkpoint_dir / 'checkpoint_best.pt'
            torch.save(checkpoint, path)
        
        path = self.checkpoint_dir / 'checkpoint_latest.pt'
        torch.save(checkpoint, path)
    
    def _cleanup_checkpoints(self):
        checkpoints = sorted(self.checkpoint_dir.glob('checkpoint_epoch_*.pt'))
        while len(checkpoints) > self.training_config.keep_last_n_checkpoints:
            checkpoints[0].unlink()
            checkpoints.pop(0)
    
    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.state.load_state_dict(checkpoint['train_state'])
        
        if self.use_amp and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.state.epoch = checkpoint['epoch']
        self.state.global_step = checkpoint['global_step']
        
        return checkpoint.get('config', {})
    
    def train(self) -> Dict[str, List]:
        print('Starting training on ' + str(self.device))
        print('Model parameters: ' + str(sum(p.numel() for p in self.model.parameters())))
        
        for epoch in range(self.state.epoch, self.training_config.max_epochs):
            self.state.epoch = epoch
            epoch_start = time.time()
            
            train_losses = self.train_epoch()
            
            val_losses = {}
            if (epoch + 1) % self.training_config.validate_every_n_epochs == 0:
                val_losses = self.validate()
            
            val_metric = val_losses.get(self.training_config.early_stopping_metric, 0.0)
            self.scheduler.step(epoch=epoch, val_metric=val_metric)
            
            if (epoch + 1) % self.pca_callback.update_every_n_epochs == 0:
                self.pca_callback.update_basis()
            
            train_losses['epoch'] = epoch
            train_losses['lr_main'] = self.optimizer.main_optimizer.param_groups[0]['lr']
            train_losses['lr_recovery'] = self.optimizer.recovery_optimizer.param_groups[0]['lr']
            train_losses['lr_adversarial'] = self.optimizer.adversarial_optimizer.param_groups[0]['lr']
            
            is_best = False
            if val_losses:
                val_losses['epoch'] = epoch
                if self.training_config.early_stopping_mode == 'min':
                    if val_metric < self.state.best_val_metric:
                        self.state.best_val_metric = val_metric
                        self.state.epochs_without_improvement = 0
                        is_best = True
                    else:
                        self.state.epochs_without_improvement += 1
                else:
                    if val_metric > self.state.best_val_metric:
                        self.state.best_val_metric = val_metric
                        self.state.epochs_without_improvement = 0
                        is_best = True
                    else:
                        self.state.epochs_without_improvement += 1
            
            self.save_checkpoint(epoch, is_best=is_best)
            
            self.state.train_history.append(train_losses)
            if val_losses:
                self.state.val_history.append(val_losses)
            
            if self.logger and hasattr(self.logger, 'info'):
                for k, v in train_losses.items():
                    self.logger.info('Epoch ' + str(epoch) + ' Train ' + k + ': ' + str(v))
                for k, v in val_losses.items():
                    self.logger.info('Epoch ' + str(epoch) + ' Val ' + k + ': ' + str(v))
            
            if epoch % 1 == 0:
                msg = 'Epoch ' + str(epoch) + '/' + str(self.training_config.max_epochs-1) + ' | Time: ' + str(time.time() - epoch_start) + 's'
                msg += ' | Train Loss: ' + str(train_losses.get('total_loss', 0))
                if val_losses:
                    msg += ' | Val Loss: ' + str(val_losses.get('total_loss', 0))
                msg += ' | Equiv Factor: ' + str(train_losses.get('equivariance_factor', 0))
                print(msg)
            
            if self.state.epochs_without_improvement >= self.training_config.early_stopping_patience:
                print('Early stopping triggered after ' + str(epoch) + ' epochs')
                break
        
        print('Training completed!')
        return {
            'train_history': self.state.train_history,
            'val_history': self.state.val_history,
        }


def Train_HWIN(
    model: HWINNet,
    train_loader,
    val_loader,
    config: Config,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, List]:
    trainer = HWINNetTrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        logger=logger,
    )
    
    return trainer.train()
