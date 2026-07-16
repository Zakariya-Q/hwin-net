"""HWIN-Net: Module 4 - Query Head (M4)"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from dataclasses import dataclass
import math

@dataclass
class QueryHeadConfig:
    head_type: str = "mlp"
    latent_dim: int = 64
    output_dim: int = 1
    hidden_dim: int = 64
    n_layers: int = 2
    output_distribution: str = "gaussian"
    dropout: float = 0.1
    activation: str = "gelu"
    norm_type: str = "layer"

class QueryHead(nn.Module):
    def __init__(self, config: QueryHeadConfig):
        super().__init__()
        self.config = config
        self.latent_dim = config.latent_dim
        self.output_dim = config.output_dim
        self.hidden_dim = config.hidden_dim
        self.n_layers = config.n_layers
        self.output_distribution = config.output_distribution
        self.head_type = config.head_type

        if config.activation == "gelu":
            act_fn = nn.GELU
        elif config.activation == "relu":
            act_fn = nn.ReLU
        elif config.activation == "silu":
            act_fn = nn.SiLU
        else:
            act_fn = nn.GELU

        def get_norm(dim:int):
            if config.norm_type == "layer":
                return nn.LayerNorm(dim)
            elif config.norm_type == "batch":
                return nn.BatchNorm1d(dim)
            else:
                return nn.Identity()

        if config.head_type == "linear":
            self.head = nn.Linear(config.latent_dim, config.output_dim)
        elif config.head_type in ("mlp", "gaussian"):
            layers = []
            in_dim = config.latent_dim
            for i in range(config.n_layers):
                out_dim = config.hidden_dim if i < config.n_layers - 1 else config.output_dim
                layers.append(nn.Linear(in_dim, out_dim))
                if i < config.n_layers - 1:
                    layers.append(get_norm(out_dim))
                    layers.append(act_fn())
                    layers.append(nn.Dropout(config.dropout))
                in_dim = out_dim
            self.head = nn.Sequential(*layers)
        else:
            raise ValueError(f"Unknown head_type: {config.head_type}")

        if config.output_distribution == "gaussian":
            var_layers = []
            in_dim = config.latent_dim
            for i in range(config.n_layers):
                out_dim = config.hidden_dim if i < config.n_layers - 1 else config.output_dim
                var_layers.append(nn.Linear(in_dim, out_dim))
                if i < config.n_layers - 1:
                    var_layers.append(get_norm(out_dim))
                    var_layers.append(act_fn())
                    var_layers.append(nn.Dropout(config.dropout))
                in_dim = out_dim
            self.var_head = nn.Sequential(*var_layers)
        else:
            self.var_head = None

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, mu_final):
        B, d = mu_final.shape
        assert d == self.latent_dim, f"Input dim {d} != latent_dim {self.latent_dim}"
        q_hat = self.head(mu_final)
        if self.output_distribution == "gaussian" and self.var_head is not None:
            log_var = self.var_head(mu_final)
            sigma2_aleat = F.softplus(log_var) + 1e-6
        else:
            sigma2_aleat = None
        return q_hat, sigma2_aleat

    def extra_repr(self):
        return f"latent_dim={self.latent_dim}, output_dim={self.output_dim}, head_type={self.head_type}, output_distribution={self.output_distribution}"

def create_query_head(config):
    return QueryHead(config)

class QueryHeadScriptable(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.latent_dim = config.latent_dim
        self.output_dim = config.output_dim
        self.output_distribution = config.output_distribution

        if config.activation == "gelu":
            act_fn = nn.GELU
        elif config.activation == "relu":
            act_fn = nn.ReLU
        elif config.activation == "silu":
            act_fn = nn.SiLU
        else:
            act_fn = nn.GELU

        def get_norm(dim:int):
            if config.norm_type == "layer":
                return nn.LayerNorm(dim)
            elif config.norm_type == "batch":
                return nn.BatchNorm1d(dim)
            else:
                return nn.Identity()

        if config.head_type == "linear":
            self.head = nn.Linear(config.latent_dim, config.output_dim)
        else:
            layers = []
            in_dim = config.latent_dim
            for i in range(config.n_layers):
                out_dim = config.hidden_dim if i < config.n_layers - 1 else config.output_dim
                layers.append(nn.Linear(in_dim, out_dim))
                if i < config.n_layers - 1:
                    layers.append(get_norm(out_dim))
                    layers.append(act_fn())
                    layers.append(nn.Dropout(config.dropout))
                in_dim = out_dim
            self.head = nn.Sequential(*layers)

        if config.output_distribution == "gaussian":
            var_layers = []
            in_dim = config.latent_dim
            for i in range(config.n_layers):
                out_dim = config.hidden_dim if i < config.n_layers - 1 else config.output_dim
                var_layers.append(nn.Linear(in_dim, out_dim))
                if i < config.n_layers - 1:
                    var_layers.append(get_norm(out_dim))
                    var_layers.append(act_fn())
                    var_layers.append(nn.Dropout(config.dropout))
                in_dim = out_dim
            self.var_head = nn.Sequential(*var_layers)
        else:
            self.var_head = None

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, mu_final):
        q_hat = self.head(mu_final)
        if self.output_distribution == "gaussian" and self.var_head is not None:
            log_var = self.var_head(mu_final)
            sigma2_aleat = F.softplus(log_var) + 1e-6
        else:
            sigma2_aleat = None
        return q_hat, sigma2_aleat

def test_query_head():
    import torch
    config = QueryHeadConfig(head_type="linear", latent_dim=64, output_dim=1, output_distribution="point")
    head = QueryHead(config)
    B = 16
    mu_final = torch.randn(B, 64)
    q_hat, sigma2 = head(mu_final)
    assert q_hat.shape == (B, 1), f"q_hat shape {q_hat.shape} != (16, 1)"
    assert sigma2 is None
    print("Test 1 PASSED: Linear head")

    config_mlp = QueryHeadConfig(head_type="mlp", latent_dim=64, output_dim=1, hidden_dim=128, n_layers=3, output_distribution="point")
    head_mlp = QueryHead(config_mlp)
    q_hat2, sigma2_2 = head_mlp(mu_final)
    assert q_hat2.shape == (B, 1)
    assert sigma2_2 is None
    print("Test 2 PASSED: MLP head")

    config_gauss = QueryHeadConfig(head_type="gaussian", latent_dim=64, output_dim=1, hidden_dim=128, n_layers=2, output_distribution="gaussian")
    head_gauss = QueryHead(config_gauss)
    q_hat3, sigma2_3 = head_gauss(mu_final)
    assert q_hat3.shape == (B, 1)
    assert sigma2_3 is not None
    assert sigma2_3.shape == (B, 1)
    assert (sigma2_3 > 0).all()
    print("Test 3 PASSED: Gaussian head")

    config_multi = QueryHeadConfig(head_type="mlp", latent_dim=64, output_dim=10, hidden_dim=128, n_layers=2, output_distribution="point")
    head_multi = QueryHead(config_multi)
    q_hat4, _ = head_multi(mu_final)
    assert q_hat4.shape == (B, 10)
    print("Test 4 PASSED: Multi-dimensional output")

    import inspect
    sig = inspect.signature(QueryHead.forward)
    params = list(sig.parameters.keys())
    assert params == ["self", "mu_final"], f"Forward should only take mu_final, got {params}"
    print("Test 5 PASSED: No schema dependence")

    head_script = QueryHeadScriptable(config_mlp)
    q_script, _ = head_script(mu_final)
    assert q_script.shape == q_hat2.shape
    print("Test 6 PASSED: Scriptable version")

    mu_final_req = mu_final.detach().requires_grad_(True)
    q_grad, _ = head_mlp(mu_final_req)
    loss = q_grad.sum()
    loss.backward()
    assert mu_final_req.grad is not None
    print("Test 7 PASSED: Gradient flow")

    print("\n=== ALL QUERY HEAD TESTS PASSED ===")

if __name__ == "__main__":
    test_query_head()
