import re

with open("C:/Users/lenovo/hwin_net/utils/config.py", "r") as f:
    content = f.read()

# Fix QueryHeadConfig
content = re.sub(r"    hidden_dim = field\(default=64, metadata=", "    hidden_dim: int = field(default=64, metadata=", content)
content = re.sub(r"    n_layers = field\(default=2, metadata=", "    n_layers: int = field(default=2, metadata=", content)

# Fix GateConfig
content = re.sub(r"    prior_predictive_type = field\(default=\"empirical_bayes\", metadata=", "    prior_predictive_type: Literal[\"empirical_bayes\", \"gaussian\", \"mixture\"] = field(default=\"empirical_bayes\", metadata=", content)
content = re.sub(r"    schemas = field\(default_factory=list, metadata=", "    schemas: List[Tuple[List[int], int]] = field(default_factory=list, metadata=", content)

# Fix NoLeakageConfig
content = re.sub(r"    discriminator_type = field\(default=\"mlp\", metadata=", "    discriminator_type: Literal[\"mlp\", \"linear\"] = field(default=\"mlp\", metadata=", content)
content = re.sub(r"    discriminator_hidden = field\(default=64, metadata=", "    discriminator_hidden: int = field(default=64, metadata=", content)
content = re.sub(r"    discriminator_layers = field\(default=2, metadata=", "    discriminator_layers: int = field(default=2, metadata=", content)
content = re.sub(r"    gradient_reversal = field\(default=True, metadata=", "    gradient_reversal: bool = field(default=True, metadata=", content)
content = re.sub(r"    grl_lambda = field\(default=1.0, metadata=", "    grl_lambda: float = field(default=1.0, metadata=", content)
content = re.sub(r"    lambda_mi = field\(default=0.1, metadata=", "    lambda_mi: float = field(default=0.1, metadata=", content)

# Fix LossConfig
content = re.sub(r"    pred_loss_type = field\(default=\"mse\", metadata=", "    pred_loss_type: Literal[\"mse\", \"mae\", \"huber\", \"nll_gaussian\", \"nll_categorical\"] = field(default=\"mse\", metadata=", content)
content = re.sub(r"    rec_loss_type = field\(default=\"mse\", metadata=", "    rec_loss_type: Literal[\"mse\", \"minmax\"] = field(default=\"mse\", metadata=", content)
content = re.sub(r"    equiv_loss_type = field\(default=\"frobenius\", metadata=", "    equiv_loss_type: Literal[\"frobenius\", \"spectral\", \"cosine\"] = field(default=\"frobenius\", metadata=", content)

with open("C:/Users/lenovo/hwin_net/utils/config.py", "w") as f:
    f.write(content)

print("All fixed")
