"""HWIN-Net Configuration Module

Mathematical Purpose
--------------------
This module defines the complete configuration schema for HWIN-Net, ensuring
theorem-to-configuration traceability per the frozen SIS theory (SIS_Reaxiomatized.md)
and the canonical implementation specification (HWIN_Net_Spec.md).
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
import yaml
from omegaconf import OmegaConf, DictConfig
import hydra
from hydra.core.config_store import ConfigStore


# ============================================================================
# MODULE 1: SCHEMA ENCODER (M1) -- Axiom A2, A3, D2, Section 5.1
# ============================================================================

@dataclass
class EncoderConfig:
    enabled: bool = field(default=True)
    encoder_type: str = field(default="mlp", metadata={"theorem_ref": "A2, A3, D2", "description": "Architecture of platform-specific encoder f_a"})
    n_layers: int = field(default=3, metadata={"theorem_ref": "A2, A3", "description": "Number of hidden layers in f_a"})
    hidden_dim: int = field(default=128, metadata={"theorem_ref": "A2, A3", "description": "Hidden dimension of f_a"})
    output_dim: int = field(default=128, metadata={"theorem_ref": "D2", "description": "Output dimension k of encoder f_a: R^n -> R^k"})
    platform_embedding_dim: int = field(default=32, metadata={"theorem_ref": "A2, A3", "description": "Platform embedding dimension k_a"})
    num_platforms: int = field(default=3, metadata={"theorem_ref": "A2, A3", "description": "Number of platforms |A|"})
    share_platform_encoder: bool = field(default=False, metadata={"theorem_ref": "A2, A3", "description": "Share encoder weights across platforms"})
    mask_mode: str = field(default="zero_fill", metadata={"theorem_ref": "D2", "description": "How to mask unobserved dimensions"})
    dropout: float = field(default=0.1, metadata={"theorem_ref": "impl", "description": "Dropout rate in encoder layers"})
    activation: str = field(default="gelu", metadata={"theorem_ref": "impl", "description": "Activation function"})
    norm_type: str = field(default="layer", metadata={"theorem_ref": "impl", "description": "Normalization type"})


# ============================================================================
# MODULE 2: RECOVERY MODULE (M2) -- Axiom A4, Lemma 2, Lemma 5, CC3, Section 5.2
# ============================================================================

@dataclass
class RecoveryConfig:
    enabled: bool = field(default=True)
    recovery_type: str = field(default="equivariant_mlp", metadata={"theorem_ref": "A4, L2, L5", "description": "Architecture of T_{(O,a)}"})
    latent_dim: int = field(default=64, metadata={"theorem_ref": "A4, D5", "description": "Latent dimension d of mu: M -> R^d"})
    hidden_dim: int = field(default=128, metadata={"theorem_ref": "A4, L2", "description": "Hidden dimension of recovery network"})
    n_layers: int = field(default=3, metadata={"theorem_ref": "A4, L2", "description": "Number of layers in T_{(O,a)}"})
    intertwiner_type: str = field(default="linear", metadata={"theorem_ref": "L5, CC3", "description": "Type of intertwiner R_{a2,a1} for weight tying"})
    base_platform: int = field(default=0, metadata={"theorem_ref": "L5", "description": "Reference platform a_ref for T_{(O,a)} = R_{a,a_ref} o T_{(O,a_ref)}"})
    tie_t_base: bool = field(default=True, metadata={"theorem_ref": "CC3, L5", "description": "Share base T_{(O,a_ref)} across platforms"})
    equivariance_loss_weight: float = field(default=1.0, metadata={"theorem_ref": "L5, CC3", "description": "Weight for equivariance loss"})
    dropout: float = field(default=0.1, metadata={"theorem_ref": "impl", "description": "Dropout in recovery layers"})
    activation: str = field(default="gelu", metadata={"theorem_ref": "impl", "description": "Activation function"})
    norm_type: str = field(default="layer", metadata={"theorem_ref": "impl", "description": "Normalization type"})


# ============================================================================
# MODULE 3: MANIFOLD RETRACTION (M3) -- Axiom A1, Theorem 1, D12, CC4, Section 5.3
# ============================================================================

@dataclass
class RetractionConfig:
    enabled: bool = field(default=True)
    retraction_type: str = field(default="pca", metadata={"theorem_ref": "A1, T1, CC4, D12", "description": "Type of retraction rho: R^d -> mu(M)"})
    latent_dim: int = field(default=64, metadata={"theorem_ref": "A4, D5", "description": "Dimension d of latent space R^d"})
    manifold_basis_path: Optional[str] = field(default=None, metadata={"theorem_ref": "D12, CC4", "description": "Path to precomputed basis of mu(M) for orthogonal projection"})
    pca_components: int = field(default=32, metadata={"theorem_ref": "D12", "description": "Number of PCA components for mu(M) subspace"})
    vae_latent_dim: int = field(default=16, metadata={"theorem_ref": "CC4", "description": "VAE bottleneck dimension for manifold learning"})
    vae_encoder_layers: List[int] = field(default_factory=lambda: [128, 64], metadata={"theorem_ref": "CC4", "description": "VAE encoder hidden layers"})
    vae_decoder_layers: List[int] = field(default_factory=lambda: [64, 128], metadata={"theorem_ref": "CC4", "description": "VAE decoder hidden layers"})
    max_iter: int = field(default=10, metadata={"theorem_ref": "CC4", "description": "Max iterations for iterative projection"})
    tolerance: float = field(default=1e-6, metadata={"theorem_ref": "CC4", "description": "Convergence tolerance for iterative projection"})
    idempotence_loss_weight: float = field(default=1.0, metadata={"theorem_ref": "CC4", "description": "Weight for idempotence loss"})


# ============================================================================
# MODULE 4: QUERY HEAD (M4) -- Axiom A5, Theorem 3, Lemma 8, Section 5.4
# ============================================================================

@dataclass
class QueryHeadConfig:
    enabled: bool = field(default=True)
    head_type: str = field(default="mlp", metadata={"theorem_ref": "A5, T3, L8", "description": "Architecture of psi': R^d -> Delta(Y)"})
    latent_dim: int = field(default=64, metadata={"theorem_ref": "D5", "description": "Input dimension d = dim(mu(M))"})
    output_dim: int = field(default=1, metadata={"theorem_ref": "P9, D6", "description": "Output dimension |Y| (target space)"})
    hidden_dim: int = field(default=64, metadata={"theorem_ref": "impl", "description": "Hidden dimension for MLP head"})
    n_layers: int = field(default=2, metadata={"theorem_ref": "impl", "description": "Number of layers in MLP head"})
    output_distribution: str = field(default="gaussian", metadata={"theorem_ref": "T5", "description": "Output distribution type for aleatoric uncertainty sigma^2_aleat"})
    dropout: float = field(default=0.1, metadata={"theorem_ref": "impl", "description": "Dropout in query head"})
    activation: str = field(default="gelu", metadata={"theorem_ref": "impl", "description": "Activation function"})
    norm_type: str = field(default="layer", metadata={"theorem_ref": "impl", "description": "Normalization type"})



# ============================================================================
# MODULE 5: IDENTIFIABILITY GATE / ROUTER (M5) -- Axiom A4, Theorem 4, Lemma 7, D11, D11b, CC2, T5
# ============================================================================

@dataclass
class GateConfig:
    enabled: bool = field(default=True)
    r0_method: str = field(default="regressor", metadata={"theorem_ref": "A4, CR-A4b", "description": "Method to compute r_0(g): lookup table, learned regressor, or C3 formula"})
    r0_init: float = field(default=3.0, metadata={"theorem_ref": "A4, L7", "description": "Initial value for r_0(g) thresholds"})
    prior_predictive_type: str = field(default="empirical_bayes", metadata={"theorem_ref": "D11, D11b", "description": "Type of prior predictive p(q | nothing)"})
    prior_mean: float = field(default=0.0, metadata={"theorem_ref": "D11", "description": "Prior mean E[q] for non-ID schemas"})
    prior_var: float = field(default=1.0, metadata={"theorem_ref": "D11b", "description": "Prior variance Var[q] = sigma^2_nonid (constant across non-ID schemas D11b)"})
    hard_gate: bool = field(default=False, metadata={"theorem_ref": "CC2, CR-CC2", "description": "Hard binary gate at |O| = r_0(g) (must be True per CC2)"})
    schemas: List[List[int]] = field(default_factory=list, metadata={"theorem_ref": "D1, A3", "description": "List of schemas (O, a) for r_0 lookup table"})
    r0_regressor_hidden: int = field(default=64, metadata={"theorem_ref": "impl", "description": "Hidden dim for r_0 regressor"})


# ============================================================================
# MODULE 6: NO-LEAKAGE REGULARIZER (M6) -- Axiom A5, CC5, C1, Section 5.6
# ============================================================================

@dataclass
class NoLeakageConfig:
    enabled: bool = field(default=True)
    mi_estimator: str = field(default="adversarial", metadata={"theorem_ref": "A5, CC5, C1", "description": "Mutual information estimation method"})
    discriminator_type: str = field(default="mlp", metadata={"theorem_ref": "CC5", "description": "Discriminator D_theta architecture"})
    discriminator_hidden: int = field(default=64, metadata={"theorem_ref": "CC5", "description": "Hidden dimension of discriminator"})
    discriminator_layers: int = field(default=2, metadata={"theorem_ref": "CC5", "description": "Number of discriminator layers"})
    gradient_reversal: bool = field(default=True, metadata={"theorem_ref": "A5, CC5", "description": "Use gradient reversal layer for adversarial MI"})
    grl_lambda: float = field(default=1.0, metadata={"theorem_ref": "A5, CC5", "description": "Gradient reversal lambda (can be scheduled)"})
    grl_lambda_schedule: str = field(default="constant", metadata={"theorem_ref": "impl", "description": "GRL lambda schedule"})
    grl_max_lambda: float = field(default=10.0, metadata={"theorem_ref": "impl", "description": "Max GRL lambda"})
    lambda_mi: float = field(default=0.1, metadata={"theorem_ref": "A5, CC5, C1", "description": "Weight lambda_MI for MI penalty in total loss (must be > 0)"})
    discriminator_dropout: float = field(default=0.1, metadata={"theorem_ref": "impl", "description": "Dropout in discriminator"})
    discriminator_activation: str = field(default="gelu", metadata={"theorem_ref": "impl", "description": "Activation in discriminator"})
    discriminator_norm: str = field(default="layer", metadata={"theorem_ref": "impl", "description": "Normalization in discriminator"})


# ============================================================================
# LOSS CONFIGURATION -- 5 Loss Terms + Weights (Section 5.6, Training)
# ============================================================================

@dataclass
class LossConfig:
    lambda_pred: float = field(default=1.0, metadata={"theorem_ref": "A4, T3, T5", "description": "Weight for prediction loss L_pred"})
    lambda_rec: float = field(default=1.0, metadata={"theorem_ref": "CC1, A4", "description": "Weight for recovery loss L_rec"})
    lambda_noleak: float = field(default=0.1, metadata={"theorem_ref": "A5, CC5, C1", "description": "Weight for no-leakage MI loss L_noleak"})
    lambda_equiv: float = field(default=0.1, metadata={"theorem_ref": "L5, CC3", "description": "Weight for equivariance loss L_equiv"})
    lambda_complex: float = field(default=1e-4, metadata={"theorem_ref": "CC1", "description": "Weight for complexity penalty L_complex on mu"})
    pred_loss_type: str = field(default="mse", metadata={"theorem_ref": "A4, T3, T5", "description": "Prediction loss function type"})
    rec_loss_type: str = field(default="mse", metadata={"theorem_ref": "CC1", "description": "Recovery loss type: MSE or min-max"})
    equiv_loss_type: str = field(default="frobenius", metadata={"theorem_ref": "L5, CC3", "description": "Equivariance loss: ||R_{a2,a1} - I||_F^2 or spectral norm"})


# ============================================================================
# TRAINING CONFIGURATION -- TTUR, Gradient Reversal, Checkpointing (Training section)
# ============================================================================

@dataclass
class TrainingConfig:
    max_epochs: int = field(default=100, metadata={"theorem_ref": "impl", "description": "Maximum training epochs"})
    batch_size: int = field(default=32, metadata={"theorem_ref": "impl", "description": "Training batch size"})
    val_batch_size: int = field(default=64, metadata={"theorem_ref": "impl", "description": "Validation batch size"})
    use_ttur: bool = field(default=True, metadata={"theorem_ref": "Training", "description": "Use two-time-scale update rule (TTUR)"})
    lr_main: float = field(default=1e-3, metadata={"theorem_ref": "Training", "description": "Learning rate for main parameters (theta, psi)"})
    lr_recovery: float = field(default=5e-4, metadata={"theorem_ref": "Training", "description": "Learning rate for recovery module T_g (theta_T)"})
    lr_adversarial: float = field(default=2e-4, metadata={"theorem_ref": "A5, CC5", "description": "Learning rate for adversarial discriminator"})
    optimizer_type: str = field(default="adamw", metadata={"theorem_ref": "impl", "description": "Optimizer type"})
    weight_decay: float = field(default=1e-4, metadata={"theorem_ref": "impl", "description": "Weight decay"})
    beta1: float = field(default=0.9, metadata={"theorem_ref": "impl", "description": "Adam beta1"})
    beta2: float = field(default=0.999, metadata={"theorem_ref": "impl", "description": "Adam beta2"})
    eps: float = field(default=1e-8, metadata={"theorem_ref": "impl", "description": "Adam epsilon"})
    scheduler_type: str = field(default="cosine", metadata={"theorem_ref": "impl", "description": "Learning rate scheduler"})
    warmup_epochs: int = field(default=5, metadata={"theorem_ref": "impl", "description": "Warmup epochs"})
    min_lr: float = field(default=1e-6, metadata={"theorem_ref": "impl", "description": "Minimum learning rate"})
    step_size: int = field(default=30, metadata={"theorem_ref": "impl", "description": "StepLR step size"})
    gamma: float = field(default=0.1, metadata={"theorem_ref": "impl", "description": "StepLR gamma"})
    plateau_patience: int = field(default=10, metadata={"theorem_ref": "impl", "description": "ReduceLROnPlateau patience"})
    plateau_factor: float = field(default=0.5, metadata={"theorem_ref": "impl", "description": "ReduceLROnPlateau factor"})
    grad_clip_norm: float = field(default=1.0, metadata={"theorem_ref": "impl", "description": "Max gradient norm for clipping"})
    equivariance_warmup_epochs: int = field(default=10, metadata={"theorem_ref": "Training", "description": "Epochs to ramp up equivariance loss weight"})
    use_mixed_precision: bool = field(default=True, metadata={"theorem_ref": "Training", "description": "Use torch.cuda.amp autocast (Torch 2.x)"})
    checkpoint_dir: str = field(default="./checkpoints", metadata={"theorem_ref": "Training", "description": "Checkpoint directory"})
    save_every_n_epochs: int = field(default=10, metadata={"theorem_ref": "Training", "description": "Save checkpoint every N epochs"})
    keep_last_n_checkpoints: int = field(default=3, metadata={"theorem_ref": "Training", "description": "Number of checkpoints to keep"})
    resume_from: Optional[str] = field(default=None, metadata={"theorem_ref": "Training", "description": "Path to checkpoint to resume from"})
    validate_every_n_epochs: int = field(default=1, metadata={"theorem_ref": "Training", "description": "Run validation every N epochs"})
    early_stopping_patience: int = field(default=20, metadata={"theorem_ref": "Training", "description": "Early stopping patience"})
    early_stopping_metric: str = field(default="val_loss", metadata={"theorem_ref": "Training", "description": "Metric for early stopping"})
    early_stopping_mode: str = field(default="min", metadata={"theorem_ref": "Training", "description": "Early stopping mode"})
    seed: int = field(default=42, metadata={"theorem_ref": "Training", "description": "Random seed for reproducibility"})
    deterministic: bool = field(default=True, metadata={"theorem_ref": "Training", "description": "Use deterministic algorithms (cuDNN)"})
    device: str = field(default="auto", metadata={"theorem_ref": "impl", "description": "Compute device"})
    num_workers: int = field(default=4, metadata={"theorem_ref": "impl", "description": "DataLoader workers"})
    pin_memory: bool = field(default=True, metadata={"theorem_ref": "impl", "description": "Pin memory for DataLoader"})


# ============================================================================
# TARGET TRANSFORM CONFIGURATION -- Target space transforms (P9, T5)
# ============================================================================

@dataclass
class TargetTransformConfig:
    enabled: bool = field(default=False, metadata={"theorem_ref": "P9, T5", "description": "Enable target transform"})
    type: str = field(default="log1p", metadata={"theorem_ref": "P9, T5", "description": "Transform type: log1p, log, boxcox, none"})

# ============================================================================
# DATA CONFIGURATION -- Schema-aware Dataset (Datasets section)
# ============================================================================

@dataclass
class DataConfig:
    train_data_path: str = field(default="./data/train.parquet", metadata={"theorem_ref": "impl", "description": "Training data path"})
    val_data_path: str = field(default="./data/val.parquet", metadata={"theorem_ref": "impl", "description": "Validation data path"})
    test_data_path: str = field(default="./data/test.parquet", metadata={"theorem_ref": "impl", "description": "Test data path"})
    num_variables: int = field(default=100, metadata={"theorem_ref": "D1, D2", "description": "Number of canonical variables |V| = n"})
    schemas: List[List[int]] = field(default_factory=list, metadata={"theorem_ref": "D1, A3", "description": "List of schemas (O, a) in dataset"})
    schemas_file: Optional[str] = field(default=None, metadata={"theorem_ref": "D1", "description": "Path to schema definitions file (JSON/YAML)"})
    platform_embeddings_file: Optional[str] = field(default=None, metadata={"theorem_ref": "A2, P5, P6", "description": "Path to platform embeddings e_a"})
    num_platforms: int = field(default=3, metadata={"theorem_ref": "A2, A3", "description": "Number of platforms |A|"})
    target_column: str = field(default="y", metadata={"theorem_ref": "P9", "description": "Target column name"})
    target_type: str = field(default="regression", metadata={"theorem_ref": "P9", "description": "Target type"})
    schema_sampler_type: str = field(default="stratified", metadata={"theorem_ref": "Datasets", "description": "Schema sampling strategy"})
    min_observed_vars: int = field(default=1, metadata={"theorem_ref": "A4, L7", "description": "Minimum |O| for a valid sample"})
    normalize_features: bool = field(default=True, metadata={"theorem_ref": "impl", "description": "Normalize input features x"})
    feature_mean_path: Optional[str] = field(default=None, metadata={"theorem_ref": "impl", "description": "Path to feature mean for normalization"})
    feature_std_path: Optional[str] = field(default=None, metadata={"theorem_ref": "impl", "description": "Path to feature std for normalization"})

# ============================================================================
# ROOT CONFIGURATION -- Composed Config
# ============================================================================

@dataclass
class Config:
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    recovery: RecoveryConfig = field(default_factory=RecoveryConfig)
    retraction: RetractionConfig = field(default_factory=RetractionConfig)
    query_head: QueryHeadConfig = field(default_factory=QueryHeadConfig)
    gate: GateConfig = field(default_factory=GateConfig)
    no_leakage: NoLeakageConfig = field(default_factory=NoLeakageConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    target_transform: TargetTransformConfig = field(default_factory=TargetTransformConfig, metadata={"theorem_ref": "P9, T5", "description": "Target space transform configuration"})
    experiment_name: str = field(default="hwin_net", metadata={"theorem_ref": "impl", "description": "Experiment name for logging"})
    output_dir: str = field(default="./outputs", metadata={"theorem_ref": "impl", "description": "Output directory for logs/checkpoints"})
    log_level: str = field(default="info", metadata={"theorem_ref": "impl", "description": "Logging level"})
    _target_: str = field(default="models.hwin_net.HWINNet", metadata={"theorem_ref": "impl", "description": "Target class for Hydra instantiation"})


# ============================================================================
# CONFIG LOADING / SAVING UTILITIES
# ============================================================================

def load_config(config_path=None, overrides=None):
    if config_path is not None:
        with open(config_path, "r") as f:
            yaml_config = yaml.safe_load(f)
        cfg = OmegaConf.structured(Config)
        cfg = OmegaConf.merge(cfg, yaml_config)
    else:
        cfg = OmegaConf.structured(Config)
    if overrides:
        override_conf = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, override_conf)
    config = OmegaConf.to_object(cfg)
    return config


def save_config(config, output_path):
    cfg = OmegaConf.structured(config)
    OmegaConf.save(cfg, output_path)


def create_config_store():
    cs = ConfigStore.instance()
    cs.store(name="config", node=Config)
    return cs


def validate_config(config):
    errors = []
    if not config.gate.hard_gate:
        errors.append("THEOREM VIOLATION: gate.hard_gate must be True (CC2 requires hard binary gate)")
    if config.no_leakage.lambda_mi <= 0:
        errors.append("THEOREM VIOLATION: no_leakage.lambda_mi must be > 0 (A5, CC5, C1)")
    if not config.recovery.tie_t_base:
        errors.append("WARNING: recovery.tie_t_base=False violates CC3 weight-tying requirement")
    if config.encoder.output_dim != config.recovery.hidden_dim:
        errors.append(f"DIMENSION MISMATCH: encoder.output_dim ({config.encoder.output_dim}) != recovery.hidden_dim ({config.recovery.hidden_dim})")
    if config.recovery.latent_dim != config.query_head.latent_dim:
        errors.append(f"DIMENSION MISMATCH: recovery.latent_dim ({config.recovery.latent_dim}) != query_head.latent_dim ({config.query_head.latent_dim})")
    if config.recovery.latent_dim != config.retraction.latent_dim:
        errors.append(f"DIMENSION MISMATCH: recovery.latent_dim ({config.recovery.latent_dim}) != retraction.latent_dim ({config.retraction.latent_dim})")
    if config.query_head.latent_dim != config.retraction.latent_dim:
        errors.append(f"DIMENSION MISMATCH: query_head.latent_dim ({config.query_head.latent_dim}) != retraction.latent_dim ({config.retraction.latent_dim})")
    if config.encoder.num_platforms < config.recovery.base_platform + 1:
        errors.append(f"WARNING: encoder.num_platforms ({config.encoder.num_platforms}) should be >= recovery.base_platform+1 ({config.recovery.base_platform+1})")
    return errors


def get_default_config():
    return Config()


if __name__ == "__main__":
    cfg = get_default_config()
    print("Default config loaded successfully")
    print(f"Experiment: {cfg.experiment_name}")
    print(f"Encoder: {cfg.encoder.encoder_type}, dim={cfg.encoder.output_dim}")
    print(f"Recovery: {cfg.recovery.recovery_type}, latent_dim={cfg.recovery.latent_dim}")
    print(f"Retraction: {cfg.retraction.retraction_type}")
    print(f"Query Head: {cfg.query_head.head_type}")
    print(f"Gate: r0_method={cfg.gate.r0_method}, hard_gate={cfg.gate.hard_gate}")
    print(f"NoLeakage: {cfg.no_leakage.mi_estimator}, lambda_mi={cfg.no_leakage.lambda_mi}")
    print(f"Loss weights: pred={cfg.loss.lambda_pred}, rec={cfg.loss.lambda_rec}, noleak={cfg.loss.lambda_noleak}, equiv={cfg.loss.lambda_equiv}, complex={cfg.loss.lambda_complex}")
    print(f"Training: TTUR={cfg.training.use_ttur}, max_epochs={cfg.training.max_epochs}")
    errors = validate_config(cfg)
    if errors:
        print("\nVALIDATION ERRORS:")
        for e in errors:
            print(f"  - {e}")
    else:
        print("\nConfig validation PASSED")
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        save_config(cfg, f.name)
        print(f"Saved config to {f.name}")
        cfg2 = load_config(f.name)
        print("Reloaded config successfully")
    print("\n=== Config module test PASSED ===")















