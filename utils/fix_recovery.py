import re

with open("C:/Users/lenovo/hwin_net/utils/config.py", "r") as f:
    content = f.read()

content = re.sub(r"    hidden_dim = field\(default=128, metadata=", "    hidden_dim: int = field(default=128, metadata=", content)
content = re.sub(r"    n_layers = field\(default=3, metadata=", "    n_layers: int = field(default=3, metadata=", content)
content = re.sub(r"    intertwiner_type = field\(default=\"linear\", metadata=", "    intertwiner_type: Literal[\"linear\", \"orthogonal\", \"shared_base\"] = field(default=\"linear\", metadata=", content)
content = re.sub(r"    base_platform = field\(default=0, metadata=", "    base_platform: int = field(default=0, metadata=", content)
content = re.sub(r"    tie_t_base = field\(default=True, metadata=", "    tie_t_base: bool = field(default=True, metadata=", content)
content = re.sub(r"    equivariance_loss_weight = field\(default=1.0, metadata=", "    equivariance_loss_weight: float = field(default=1.0, metadata=", content)
content = re.sub(r"    dropout = field\(default=0.1, metadata=", "    dropout: float = field(default=0.1, metadata=", content)
content = re.sub(r"    activation = field\(default=\"gelu\", metadata=", "    activation: Literal[\"relu\", \"gelu\", \"silu\", \"tanh\"] = field(default=\"gelu\", metadata=", content)
content = re.sub(r"    norm_type = field\(default=\"layer\", metadata=", "    norm_type: Literal[\"layer\", \"batch\", \"none\"] = field(default=\"layer\", metadata=", content)

with open("C:/Users/lenovo/hwin_net/utils/config.py", "w") as f:
    f.write(content)

print("Recovery fixed")
