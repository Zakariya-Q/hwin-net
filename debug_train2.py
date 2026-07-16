import sys
sys.path.insert(0, 'C:/Users/lenovo/hwin_net')
import torch
from models.hwin_net import create_hwin_net
from utils.config import get_default_config

config = get_default_config()
config.loss.lambda_equiv = 0.01
config.loss.lambda_noleak = 0.01
config.loss.lambda_complex = 1e-6
config.training.equivariance_warmup_epochs = 0
config.training.lr_main = 0.01

model = create_hwin_net(config)
model.train()

n_vars = config.data.num_variables
W_true = torch.randn(n_vars)
W_true[2:] = 0

for epoch in range(5):
    B = 32
    x = torch.randn(B, n_vars) * 2.0
    y_dense = x @ W_true + 0.1 * torch.randn(B)
    
    M_O = torch.zeros(B, n_vars)
    for i in range(B):
        n_obs = 4
        obs_idx = torch.randperm(n_vars)[:n_obs]
        M_O[i, obs_idx] = 1.0
    x_sparse = x * M_O
    a_idx = torch.zeros(B, dtype=torch.long)
    
    out = model.forward(x_sparse, M_O, a_idx, y=y_dense, training=True)
    loss = out['losses']['total_loss']
    
    print("Epoch {}: total={:.4f}".format(epoch, loss.item()))
    for k, v in out['losses'].items():
        if hasattr(v, 'item'):
            print("  {}: {:.4f}".format(k, v.item()))
        else:
            print("  {}: {:.4f}".format(k, v))
