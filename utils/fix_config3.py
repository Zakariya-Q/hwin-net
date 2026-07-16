import re

with open("C:/Users/lenovo/hwin_net/utils/config.py", "r") as f:
    content = f.read()

# TrainingConfig
content = re.sub(r'    max_epochs = field\(default=100, metadata=',
    '    max_epochs: int = field(default=100, metadata=', content)

content = re.sub(r'    batch_size = field\(default=32, metadata=',
    '    batch_size: int = field(default=32, metadata=', content)

content = re.sub(r'    val_batch_size = field\(default=64, metadata=',
    '    val_batch_size: int = field(default=64, metadata=', content)

content = re.sub(r'    use_ttur = field\(default=True, metadata=',
    '    use_ttur: bool = field(default=True, metadata=', content)

content = re.sub(r'    lr_main = field\(default=1e-3, metadata=',
    '    lr_main: float = field(default=1e-3, metadata=', content)

content = re.sub(r'    lr_recovery = field\(default=5e-4, metadata=',
    '    lr_recovery: float = field(default=5e-4, metadata=', content)

content = re.sub(r'    lr_adversarial = field\(default=2e-4, metadata=',
    '    lr_adversarial: float = field(default=2e-4, metadata=', content)

content = re.sub(r'    optimizer_type = field\(default="adamw", metadata=',
    r'    optimizer_type: Literal["adam", "adamw", "sgd"] = field(default="adamw", metadata=', content)

content = re.sub(r'    weight_decay = field\(default=1e-4, metadata=',
    '    weight_decay: float = field(default=1e-4, metadata=', content)

content = re.sub(r'    beta1 = field\(default=0.9, metadata=',
    '    beta1: float = field(default=0.9, metadata=', content)

content = re.sub(r'    beta2 = field\(default=0.999, metadata=',
    '    beta2: float = field(default=0.999, metadata=', content)

content = re.sub(r'    eps = field\(default=1e-8, metadata=',
    '    eps: float = field(default=1e-8, metadata=', content)

content = re.sub(r'    scheduler_type = field\(default="cosine", metadata=',
    r'    scheduler_type: Literal["cosine", "step", "reduce_on_plateau", "none"] = field(default="cosine", metadata=', content)

content = re.sub(r'    warmup_epochs = field\(default=5, metadata=',
    '    warmup_epochs: int = field(default=5, metadata=', content)

content = re.sub(r'    min_lr = field\(default=1e-6, metadata=',
    '    min_lr: float = field(default=1e-6, metadata=', content)

content = re.sub(r'    step_size = field\(default=30, metadata=',
    '    step_size: int = field(default=30, metadata=', content)

content = re.sub(r'    gamma = field\(default=0.1, metadata=',
    '    gamma: float = field(default=0.1, metadata=', content)

content = re.sub(r'    plateau_patience = field\(default=10, metadata=',
    '    plateau_patience: int = field(default=10, metadata=', content)

content = re.sub(r'    plateau_factor = field\(default=0.5, metadata=',
    '    plateau_factor: float = field(default=0.5, metadata=', content)

content = re.sub(r'    grad_clip_norm = field\(default=1.0, metadata=',
    '    grad_clip_norm: float = field(default=1.0, metadata=', content)

content = re.sub(r'    use_mixed_precision = field\(default=True, metadata=',
    '    use_mixed_precision: bool = field(default=True, metadata=', content)

content = re.sub(r'    checkpoint_dir = field\(default="\./checkpoints", metadata=',
    '    checkpoint_dir: str = field(default="./checkpoints", metadata=', content)

content = re.sub(r'    save_every_n_epochs = field\(default=10, metadata=',
    '    save_every_n_epochs: int = field(default=10, metadata=', content)

content = re.sub(r'    keep_last_n_checkpoints = field\(default=3, metadata=',
    '    keep_last_n_checkpoints: int = field(default=3, metadata=', content)

content = re.sub(r'    resume_from = field\(default=None, metadata=',
    '    resume_from: Optional[str] = field(default=None, metadata=', content)

content = re.sub(r'    validate_every_n_epochs = field\(default=1, metadata=',
    '    validate_every_n_epochs: int = field(default=1, metadata=', content)

content = re.sub(r'    early_stopping_patience = field\(default=20, metadata=',
    '    early_stopping_patience: int = field(default=20, metadata=', content)

content = re.sub(r'    early_stopping_metric = field\(default="val_loss", metadata=',
    '    early_stopping_metric: str = field(default="val_loss", metadata=', content)

content = re.sub(r'    early_stopping_mode = field\(default="min", metadata=',
    r'    early_stopping_mode: Literal["min", "max"] = field(default="min", metadata=', content)

content = re.sub(r'    seed = field\(default=42, metadata=',
    '    seed: int = field(default=42, metadata=', content)

content = re.sub(r'    deterministic = field\(default=True, metadata=',
    '    deterministic: bool = field(default=True, metadata=', content)

content = re.sub(r'    device = field\(default="auto", metadata=',
    r'    device: Literal["auto", "cpu", "cuda", "mps"] = field(default="auto", metadata=', content)

content = re.sub(r'    num_workers = field\(default=4, metadata=',
    '    num_workers: int = field(default=4, metadata=', content)

content = re.sub(r'    pin_memory = field\(default=True, metadata=',
    '    pin_memory: bool = field(default=True, metadata=', content)

# DataConfig
content = re.sub(r'    train_data_path = field\(default="\./data/train\.parquet", metadata=',
    '    train_data_path: str = field(default="./data/train.parquet", metadata=', content)

content = re.sub(r'    val_data_path = field\(default="\./data/val\.parquet", metadata=',
    '    val_data_path: str = field(default="./data/val.parquet", metadata=', content)

content = re.sub(r'    test_data_path = field\(default="\./data/test\.parquet", metadata=',
    '    test_data_path: str = field(default="./data/test.parquet", metadata=', content)

content = re.sub(r'    num_variables = field\(default=100, metadata=',
    '    num_variables: int = field(default=100, metadata=', content)

content = re.sub(r'    schemas_file = field\(default=None, metadata=',
    '    schemas_file: Optional[str] = field(default=None, metadata=', content)

content = re.sub(r'    platform_embeddings_file = field\(default=None, metadata=',
    '    platform_embeddings_file: Optional[str] = field(default=None, metadata=', content)

content = re.sub(r'    num_platforms = field\(default=3, metadata=',
    '    num_platforms: int = field(default=3, metadata=', content)

content = re.sub(r'    target_column = field\(default="y", metadata=',
    '    target_column: str = field(default="y", metadata=', content)

content = re.sub(r'    target_type = field\(default="regression", metadata=',
    r'    target_type: Literal["regression", "classification"] = field(default="regression", metadata=', content)

content = re.sub(r'    schema_sampler_type = field\(default="stratified", metadata=',
    r'    schema_sampler_type: Literal["uniform", "stratified", "importance"] = field(default="stratified", metadata=', content)

content = re.sub(r'    min_observed_vars = field\(default=1, metadata=',
    '    min_observed_vars: int = field(default=1, metadata=', content)

content = re.sub(r'    normalize_features = field\(default=True, metadata=',
    '    normalize_features: bool = field(default=True, metadata=', content)

content = re.sub(r'    feature_mean_path = field\(default=None, metadata=',
    '    feature_mean_path: Optional[str] = field(default=None, metadata=', content)

content = re.sub(r'    feature_std_path = field\(default=None, metadata=',
    '    feature_std_path: Optional[str] = field(default=None, metadata=', content)

# Root Config
content = re.sub(r'    encoder = field\(default_factory=EncoderConfig\)',
    '    encoder: EncoderConfig = field(default_factory=EncoderConfig)', content)

content = re.sub(r'    recovery = field\(default_factory=RecoveryConfig\)',
    '    recovery: RecoveryConfig = field(default_factory=RecoveryConfig)', content)

content = re.sub(r'    retraction = field\(default_factory=RetractionConfig\)',
    '    retraction: RetractionConfig = field(default_factory=RetractionConfig)', content)

content = re.sub(r'    query_head = field\(default_factory=QueryHeadConfig\)',
    '    query_head: QueryHeadConfig = field(default_factory=QueryHeadConfig)', content)

content = re.sub(r'    gate = field\(default_factory=GateConfig\)',
    '    gate: GateConfig = field(default_factory=GateConfig)', content)

content = re.sub(r'    no_leakage = field\(default_factory=NoLeakageConfig\)',
    '    no_leakage: NoLeakageConfig = field(default_factory=NoLeakageConfig)', content)

content = re.sub(r'    loss = field\(default_factory=LossConfig\)',
    '    loss: LossConfig = field(default_factory=LossConfig)', content)

content = re.sub(r'    training = field\(default_factory=TrainingConfig\)',
    '    training: TrainingConfig = field(default_factory=TrainingConfig)', content)

content = re.sub(r'    data = field\(default_factory=DataConfig\)',
    '    data: DataConfig = field(default_factory=DataConfig)', content)

content = re.sub(r'    experiment_name = field\(default="hwin_net", metadata=',
    '    experiment_name: str = field(default="hwin_net", metadata=', content)

content = re.sub(r'    output_dir = field\(default="\./outputs", metadata=',
    '    output_dir: str = field(default="./outputs", metadata=', content)

content = re.sub(r'    log_level = field\(default="info", metadata=',
    r'    log_level: Literal["debug", "info", "warning", "error"] = field(default="info", metadata=', content)

content = re.sub(r'    _target_ = field\(default="models\.hwin_net\.HWINNet", metadata=',
    '    _target_: str = field(default="models.hwin_net.HWINNet", metadata=', content)

with open("C:/Users/lenovo/hwin_net/utils/config.py", "w") as f:
    f.write(content)

print("Done with Training, Data, and Root Config")
