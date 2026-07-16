import sys
sys.path.insert(0, "C:/Users/lenovo/hwin_net")
import torch
from models.hwin_net import create_hwin_net
from utils.config import get_default_config

config = get_default_config()
config.loss.lambda_equiv = 0.01
config.loss.lambda_noleak = 0.01
config.loss.lambda_complex = 1e-6
config.training.equivariance_warmup_epochs = 0

model = create_hwin_net(config)
model.train()

n_vars = config.data.num_variables
W_true = torch.randn(n_vars)
W_true[2:] = 0

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-5)

for epoch in range(100):
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
    loss = out["losses"]["total_loss"]
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    
    if epoch % 10 == 0:
        print("Epoch {}: total={:.4f}, pred={:.4f}, lr={:.6f}".format(epoch, loss.item(), out["losses"]["pred_loss"].item(), scheduler.get_last_lr()[0]))

print()
print("Final evaluation:")
model.eval()
with torch.no_grad():
    B = 200
    x = torch.randn(B, n_vars) * 2.0
    y_dense = x @ W_true + 0.1 * torch.randn(B)
    M_O = torch.zeros(B, n_vars)
    for i in range(B):
        n_obs = 4
        obs_idx = torch.randperm(n_vars)[:n_obs]
        M_O[i, obs_idx] = 1.0
    x_sparse = x * M_O
    a_idx = torch.zeros(B, dtype=torch.long)
    
    out = model.forward(x_sparse, M_O, a_idx, y=y_dense, training=False)
    pred = out["q_out"]
    
    mse = torch.nn.functional.mse_loss(pred, y_dense).item()
    ss_res = ((y_dense - pred) ** 2).sum().item()
    ss_tot = ((y_dense - y_dense.mean()) ** 2).sum().item()
    r2 = 1 - ss_res / ss_tot
    print("MSE: {:.4f}, R2: {:.4f}".format(mse, r2))
    print("Routed: {:.4f}".format(out["routed"].mean().item()))
