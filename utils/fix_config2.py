import re

with open("C:/Users/lenovo/hwin_net/utils/config.py", "r") as f:
    content = f.read()

# Module 2: RecoveryConfig
content = re.sub(r'    recovery_type = field\(default="equivariant_mlp", metadata=',
    r'    recovery_type: Literal["equivariant_mlp", "transformer", "tie_weights"] = field(default="equivariant_mlp", metadata=', content)

content = re.sub(r'    latent_dim = field\(default=64, metadata=',
    '    latent_dim: int = field(default=64, metadata=', content)

content = re.sub(r'    hidden_dim = field\(default=128, metadata=',
    '    hidden_dim: int = field(default=128, metadata=', content)

content = re.sub(r'    n_layers = field\(default=3, metadata=',
    '    n_layers: int = field(default=3, metadata=', content)

content = re.sub(r'    intertwiner_type = field\(default="linear", metadata=',
    r'    intertwiner_type: Literal["linear", "orthogonal", "shared_base"] = field(default="linear", metadata=', content)

content = re.sub(r'    base_platform = field\(default=0, metadata=',
    '    base_platform: int = field(default=0, metadata=', content)

content = re.sub(r'    tie_t_base = field\(default=True, metadata=',
    '    tie_t_base: bool = field(default=True, metadata=', content)

content = re.sub(r'    equivariance_loss_weight = field\(default=1.0, metadata=',
    '    equivariance_loss_weight: float = field(default=1.0, metadata=', content)

# Module 3: RetractionConfig
content = re.sub(r'    retraction_type = field\(default="pca", metadata=',
    r'    retraction_type: Literal["pca", "vae", "iterative", "orthogonal", "riemannian"] = field(default="pca", metadata=', content)

content = re.sub(r'    manifold_basis_path = field\(default=None, metadata=',
    '    manifold_basis_path: Optional[str] = field(default=None, metadata=', content)

content = re.sub(r'    pca_components = field\(default=32, metadata=',
    '    pca_components: int = field(default=32, metadata=', content)

content = re.sub(r'    vae_latent_dim = field\(default=16, metadata=',
    '    vae_latent_dim: int = field(default=16, metadata=', content)

content = re.sub(r'    vae_encoder_layers = field\(default_factory=lambda: \[128, 64\], metadata=',
    '    vae_encoder_layers: List[int] = field(default_factory=lambda: [128, 64], metadata=', content)

content = re.sub(r'    vae_decoder_layers = field\(default_factory=lambda: \[64, 128\], metadata=',
    '    vae_decoder_layers: List[int] = field(default_factory=lambda: [64, 128], metadata=', content)

content = re.sub(r'    max_iter = field\(default=10, metadata=',
    '    max_iter: int = field(default=10, metadata=', content)

content = re.sub(r'    tolerance = field\(default=1e-6, metadata=',
    '    tolerance: float = field(default=1e-6, metadata=', content)

content = re.sub(r'    idempotence_loss_weight = field\(default=1.0, metadata=',
    '    idempotence_loss_weight: float = field(default=1.0, metadata=', content)

# Module 4: QueryHeadConfig
content = re.sub(r'    head_type = field\(default="mlp", metadata=',
    r'    head_type: Literal["linear", "mlp", "gp", "bayesian"] = field(default="mlp", metadata=', content)

content = re.sub(r'    output_dim = field\(default=1, metadata=',
    '    output_dim: int = field(default=1, metadata=', content)

content = re.sub(r'    n_layers = field\(default=2, metadata=',
    '    n_layers: int = field(default=2, metadata=', content)

content = re.sub(r'    output_distribution = field\(default="gaussian", metadata=',
    r'    output_distribution: Literal["point", "gaussian", "categorical"] = field(default="gaussian", metadata=', content)

# Module 5: GateConfig
content = re.sub(r'    r0_method = field\(default="lookup", metadata=',
    r'    r0_method: Literal["lookup", "regressor", "smooth_c3"] = field(default="lookup", metadata=', content)

content = re.sub(r'    r0_init = field\(default=10.0, metadata=',
    '    r0_init: float = field(default=10.0, metadata=', content)

content = re.sub(r'    prior_predictive_type = field\(default="empirical_bayes", metadata=',
    r'    prior_predictive_type: Literal["empirical_bayes", "gaussian", "mixture"] = field(default="empirical_bayes", metadata=', content)

content = re.sub(r'    prior_mean = field\(default=0.0, metadata=',
    '    prior_mean: float = field(default=0.0, metadata=', content)

content = re.sub(r'    prior_var = field\(default=1.0, metadata=',
    '    prior_var: float = field(default=1.0, metadata=', content)

content = re.sub(r'    hard_gate = field\(default=True, metadata=',
    '    hard_gate: bool = field(default=True, metadata=', content)

content = re.sub(r'    schemas = field\(default_factory=list, metadata=',
    '    schemas: List[Tuple[List[int], int]] = field(default_factory=list, metadata=', content)

content = re.sub(r'    r0_regressor_hidden = field\(default=64, metadata=',
    '    r0_regressor_hidden: int = field(default=64, metadata=', content)

# Module 6: NoLeakageConfig
content = re.sub(r'    mi_estimator = field\(default="adversarial", metadata=',
    r'    mi_estimator: Literal["adversarial", "mine", "hsic", "infonce", "mmd"] = field(default="adversarial", metadata=', content)

content = re.sub(r'    discriminator_type = field\(default="mlp", metadata=',
    r'    discriminator_type: Literal["mlp", "linear"] = field(default="mlp", metadata=', content)

content = re.sub(r'    discriminator_hidden = field\(default=64, metadata=',
    '    discriminator_hidden: int = field(default=64, metadata=', content)

content = re.sub(r'    discriminator_layers = field\(default=2, metadata=',
    '    discriminator_layers: int = field(default=2, metadata=', content)

content = re.sub(r'    gradient_reversal = field\(default=True, metadata=',
    '    gradient_reversal: bool = field(default=True, metadata=', content)

content = re.sub(r'    grl_lambda = field\(default=1.0, metadata=',
    '    grl_lambda: float = field(default=1.0, metadata=', content)

content = re.sub(r'    lambda_mi = field\(default=0.1, metadata=',
    '    lambda_mi: float = field(default=0.1, metadata=', content)

# LossConfig
content = re.sub(r'    lambda_pred = field\(default=1.0, metadata=',
    '    lambda_pred: float = field(default=1.0, metadata=', content)

content = re.sub(r'    lambda_rec = field\(default=1.0, metadata=',
    '    lambda_rec: float = field(default=1.0, metadata=', content)

content = re.sub(r'    lambda_noleak = field\(default=0.1, metadata=',
    '    lambda_noleak: float = field(default=0.1, metadata=', content)

content = re.sub(r'    lambda_equiv = field\(default=0.1, metadata=',
    '    lambda_equiv: float = field(default=0.1, metadata=', content)

content = re.sub(r'    lambda_complex = field\(default=1e-4, metadata=',
    '    lambda_complex: float = field(default=1e-4, metadata=', content)

content = re.sub(r'    pred_loss_type = field\(default="mse", metadata=',
    r'    pred_loss_type: Literal["mse", "mae", "huber", "nll_gaussian", "nll_categorical"] = field(default="mse", metadata=', content)

content = re.sub(r'    rec_loss_type = field\(default="mse", metadata=',
    r'    rec_loss_type: Literal["mse", "minmax"] = field(default="mse", metadata=', content)

content = re.sub(r'    equiv_loss_type = field\(default="frobenius", metadata=',
    r'    equiv_loss_type: Literal["frobenius", "spectral", "cosine"] = field(default="frobenius", metadata=', content)

with open("C:/Users/lenovo/hwin_net/utils/config.py", "w") as f:
    f.write(content)

print("Done with modules 2-6 and LossConfig")
