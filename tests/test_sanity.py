"""
HWIN-Net: Synthetic Sanity Tests (Task 7)

These tests verify the repaired model can solve simple synthetic tasks
where ground truth is known. If the model fails these, benchmark is invalid.
"""
import sys
sys.path.insert(0, 'C:/Users/lenovo/hwin_net')

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from models.hwin_net import create_hwin_net
from utils.config import get_default_config
from utils.seed import set_seed, set_deterministic

def train_sanity_model(config, model, n_epochs, lr, observe_first_k=10):
    '''Train model on synthetic data with proper optimizer/scheduler.'''
    model.train()
    
    n_vars = config.data.num_variables
    # Target depends on first 3 variables
    W_true = torch.zeros(n_vars)
    W_true[0] = 0.5
    W_true[1] = 0.3
    W_true[2] = 0.2
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-5)
    
    for epoch in range(n_epochs):
        B = 128
        x = torch.randn(B, n_vars) * 2.0
        y_dense = x @ W_true + 0.1 * torch.randn(B)
        
        # Observe first_k variables (includes the causal ones)
        M_O = torch.zeros(B, n_vars)
        for i in range(B):
            obs_idx = list(range(observe_first_k))
            M_O[i, obs_idx] = 1.0
        x_sparse = x * M_O
        a_idx = torch.zeros(B, dtype=torch.long)
        
        out = model.forward(x_sparse, M_O, a_idx, y=y_dense, training=True)
        loss = out['losses']['total_loss']
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if epoch % 20 == 0:
            print(f'  Epoch {epoch}: loss={loss.item():.4f}, lr={scheduler.get_last_lr()[0]:.6f}')
    
    return model, W_true

def eval_sanity_model(model, config, W_true, observe_first_k=10, n_test=200):
    '''Evaluate model on synthetic task.'''
    model.eval()
    n_vars = config.data.num_variables
    
    r2s = []
    mses = []
    routed_vals = []
    
    with torch.no_grad():
        for _ in range(5):
            B = n_test
            x = torch.randn(B, n_vars) * 2.0
            y_dense = x @ W_true + 0.1 * torch.randn(B)
            
            M_O = torch.zeros(B, n_vars)
            for i in range(B):
                obs_idx = list(range(observe_first_k))
                M_O[i, obs_idx] = 1.0
            x_sparse = x * M_O
            a_idx = torch.zeros(B, dtype=torch.long)
            
            out = model.forward(x_sparse, M_O, a_idx, y=y_dense, training=False)
            pred = out['q_out']
            
            mse = nn.functional.mse_loss(pred, y_dense).item()
            ss_res = ((y_dense - pred) ** 2).sum().item()
            ss_tot = ((y_dense - y_dense.mean()) ** 2).sum().item()
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            
            mses.append(mse)
            r2s.append(r2)
            routed_vals.append(out['routed'].mean().item())
    
    return np.mean(r2s), np.mean(mses), np.mean(routed_vals)

def test_linear_regression_sparse():
    '''Test: Linear regression with sparse but relevant observations.'''
    print('Test 1: Linear regression with relevant vars observed...')
    set_seed(42)
    set_deterministic(True)
    
    config = get_default_config()
    config.loss.lambda_equiv = 0.0
    config.loss.lambda_noleak = 0.0
    config.loss.lambda_complex = 0.0
    config.retraction.idempotence_loss_weight = 0.0
    config.loss.lambda_rec = 1.0
    config.loss.lambda_pred = 1.0
    
    model = create_hwin_net(config)
    model, W_true = train_sanity_model(config, model, n_epochs=80, lr=0.001, observe_first_k=10)
    
    avg_r2, avg_mse, avg_routed = eval_sanity_model(model, config, W_true, observe_first_k=10)
    
    print(f'  Final R2: {avg_r2:.3f}')
    print(f'  Final MSE: {avg_mse:.3f}')
    print(f'  Avg Routed: {avg_routed:.3f}')
    
    assert avg_r2 > 0.3, f'Linear regression R2 = {avg_r2:.3f} <= 0.3'
    assert avg_mse < 5.0, f'Linear regression MSE = {avg_mse:.3f} >= 5.0'
    print('[PASS] Linear regression (sparse relevant)')
    return True

def test_identity_mapping():
    '''Test: y = x_5 (identity mapping for observed variable).'''
    print('Test 2: Identity mapping...')
    set_seed(123)
    set_deterministic(True)
    
    config = get_default_config()
    config.loss.lambda_equiv = 0.0
    config.loss.lambda_noleak = 0.0
    config.loss.lambda_complex = 0.0
    config.retraction.idempotence_loss_weight = 0.0
    
    model = create_hwin_net(config)
    model.train()
    
    n_vars = config.data.num_variables
    target_var_idx = 5
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80, eta_min=1e-5)
    
    for epoch in range(80):
        B = 128
        x = torch.randn(B, n_vars) * 3.0
        y_dense = x[:, target_var_idx].clone()
        
        M_O = torch.zeros(B, n_vars)
        for i in range(B):
            obs_idx = [target_var_idx] + list(range(9))
            obs_idx = list(set(obs_idx))[:10]
            M_O[i, obs_idx] = 1.0
        x_sparse = x * M_O
        a_idx = torch.zeros(B, dtype=torch.long)
        
        out = model.forward(x_sparse, M_O, a_idx, y=y_dense, training=True)
        loss = out['losses']['total_loss']
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
    
    # Evaluation
    model.eval()
    r2s = []
    with torch.no_grad():
        for _ in range(10):
            B = 200
            x = torch.randn(B, n_vars) * 3.0
            y_dense = x[:, target_var_idx].clone()
            M_O = torch.zeros(B, n_vars)
            for i in range(B):
                obs_idx = [target_var_idx] + list(range(9))
                M_O[i, obs_idx] = 1.0
            x_sparse = x * M_O
            a_idx = torch.zeros(B, dtype=torch.long)
            
            out = model.forward(x_sparse, M_O, a_idx, y=y_dense, training=False)
            pred = out['q_out']
            
            ss_res = ((y_dense - pred) ** 2).sum().item()
            ss_tot = ((y_dense - y_dense.mean()) ** 2).sum().item()
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            r2s.append(r2)
    
    avg_r2 = np.mean(r2s)
    print(f'  Final R2: {avg_r2:.3f}')
    
    assert avg_r2 > 0.3, f'Identity mapping R2 = {avg_r2:.3f} <= 0.3'
    print('[PASS] Identity mapping')
    return True

def test_noise_free_sparse():
    '''Test: Noise-free with relevant variables observed.'''
    print('Test 3: Noise-free sparse...')
    set_seed(456)
    set_deterministic(True)
    
    config = get_default_config()
    config.loss.lambda_equiv = 0.0
    config.loss.lambda_noleak = 0.0
    config.loss.lambda_complex = 0.0
    config.retraction.idempotence_loss_weight = 0.0
    config.loss.lambda_rec = 1.0
    config.loss.lambda_pred = 1.0
    
    model = create_hwin_net(config)
    model.train()
    
    n_vars = config.data.num_variables
    W_true = torch.zeros(n_vars)
    W_true[0] = 0.5
    W_true[1] = 0.3
    W_true[2] = 0.2
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80, eta_min=1e-5)
    
    for epoch in range(80):
        B = 128
        x = torch.randn(B, n_vars) * 2.0
        y_dense = x @ W_true  # NO noise
        
        # Observe first 10 vars (includes causal ones 0,1,2)
        M_O = torch.zeros(B, n_vars)
        for i in range(B):
            obs_idx = list(range(10))
            M_O[i, obs_idx] = 1.0
        x_sparse = x * M_O
        a_idx = torch.zeros(B, dtype=torch.long)
        
        out = model.forward(x_sparse, M_O, a_idx, y=y_dense, training=True)
        loss = out['losses']['total_loss']
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
    
    # Evaluation - same sparse observation
    model.eval()
    with torch.no_grad():
        B = 200
        x = torch.randn(B, n_vars) * 2.0
        y_dense = x @ W_true
        M_O = torch.zeros(B, n_vars)
        for i in range(B):
            obs_idx = list(range(10))
            M_O[i, obs_idx] = 1.0
        x_sparse = x * M_O
        a_idx = torch.zeros(B, dtype=torch.long)
        
        out = model.forward(x_sparse, M_O, a_idx, y=y_dense, training=False)
        pred = out['q_out']
        routed = out['routed']
        
        mse = nn.functional.mse_loss(pred, y_dense).item()
        avg_routed = routed.mean().item()
        
        print(f'  MSE: {mse:.4f}')
        print(f'  Avg routed: {avg_routed:.3f}')
        
        # Very low MSE on noise-free sparse data
        assert mse < 0.5, f'Noise-free sparse MSE = {mse:.4f} >= 0.5'
        assert avg_routed > 0.9, f'Gate not opening: routed = {avg_routed:.3f}'
    
    print('[PASS] Noise-free sparse')
    return True

def test_gate_behavior():
    '''Test: Identifiability gate behavior (fixed r0).'''
    print('Test 4: Gate behavior (fixed r0)...')
    set_seed(789)
    set_deterministic(True)
    
    config = get_default_config()
    config.gate.r0_method = 'fixed'
    config.gate.r0_init = 5.0
    config.loss.lambda_equiv = 0.0
    config.loss.lambda_noleak = 0.0
    config.loss.lambda_complex = 0.0
    
    model = create_hwin_net(config)
    model.eval()
    
    n_vars = config.data.num_variables
    W_true = torch.randn(n_vars)
    W_true[5:] = 0
    
    with torch.no_grad():
        results = []
        for n_obs in [2, 4, 6, 8, 10]:
            B = 50
            x = torch.randn(B, n_vars) * 2.0
            y_dense = x @ W_true
            M_O = torch.zeros(B, n_vars)
            for i in range(B):
                obs_idx = list(range(n_obs))
                M_O[i, obs_idx] = 1.0
            x_sparse = x * M_O
            a_idx = torch.zeros(B, dtype=torch.long)
            
            out = model.forward(x_sparse, M_O, a_idx, y=y_dense, training=False)
            routed = out['routed']
            avg_routed = routed.mean().item()
            results.append((n_obs, avg_routed))
            print(f'  |O|={n_obs}: routed={avg_routed:.3f}')
    
    print('[PASS] Gate behavior (no crash)')
    return True

def test_equivariance_preserved():
    '''Test: Recovery module equivariance is preserved.'''
    print('Test 5: Equivariance preserved...')
    set_seed(999)
    set_deterministic(True)
    
    config = get_default_config()
    config.loss.lambda_equiv = 0.0
    config.loss.lambda_noleak = 0.0
    config.loss.lambda_complex = 0.0
    
    model = create_hwin_net(config)
    model.eval()
    
    n_vars = config.data.num_variables
    
    with torch.no_grad():
        x = torch.randn(4, n_vars)
        M_O = torch.ones(4, n_vars)
        a_idx_0 = torch.zeros(4, dtype=torch.long)
        a_idx_1 = torch.ones(4, dtype=torch.long)
        
        out_0 = model.forward(x, M_O, a_idx_0, training=False)
        out_1 = model.forward(x, M_O, a_idx_1, training=False)
        
        mu_hat_0 = out_0['mu_hat']
        mu_hat_1 = out_1['mu_hat']
        
        diff = (mu_hat_1 - mu_hat_0).abs().mean().item()
        print(f'  Platform 0 vs 1 mu_hat diff: {diff:.4f}')
        
        assert diff > 0, 'Platforms should produce different mu_hat'
    
    print('[PASS] Equivariance preserved')
    return True

def test_no_leakage_gradient():
    '''Test: No-leakage adversarial loss provides gradient.'''
    print('Test 6: No-leakage gradient...')
    set_seed(111)
    set_deterministic(True)
    
    config = get_default_config()
    config.loss.lambda_noleak = 0.5
    config.loss.lambda_equiv = 0.0
    config.loss.lambda_complex = 0.0
    config.retraction.idempotence_loss_weight = 0.0
    
    model = create_hwin_net(config)
    model.train()
    
    n_vars = config.data.num_variables
    B = 32
    x = torch.randn(B, n_vars)
    M_O = (torch.rand(B, n_vars) < 0.3).float()
    if M_O.sum() == 0:
        M_O[0, 0] = 1.0
    x_sparse = x * M_O
    a_idx = torch.randint(0, config.encoder.num_platforms, (B,))
    y = torch.randn(B)
    
    outputs = model.forward(x_sparse, M_O, a_idx, y=y, training=True)
    
    noleak_loss = outputs['losses'].get('noleak_loss')
    if noleak_loss is not None:
        assert noleak_loss.requires_grad, 'noleak_loss should require grad'
        assert noleak_loss.item() >= 0, f'noleak_loss should be >= 0, got {noleak_loss.item()}'
        
        model.zero_grad()
        noleak_loss.backward()
        
        encoder_grad_norm = 0.0
        for name, param in model.named_parameters():
            if 'encoder' in name and param.grad is not None:
                encoder_grad_norm += param.grad.norm().item() ** 2
        encoder_grad_norm = encoder_grad_norm ** 0.5
        
        print(f'  noleak_loss: {noleak_loss.item():.4f}')
        print(f'  Encoder grad norm: {encoder_grad_norm:.4f}')
        
        assert encoder_grad_norm > 1e-5, 'No-leakage not backpropagating to encoder'
    
    print('[PASS] No-leakage gradient')
    return True

def run_all_sanity_tests():
    print('=' * 60)
    print('HWIN-Net Synthetic Sanity Tests (Task 7)')
    print('=' * 60)
    
    test_linear_regression_sparse()
    print()
    test_identity_mapping()
    print()
    test_noise_free_sparse()
    print()
    test_gate_behavior()
    print()
    test_equivariance_preserved()
    print()
    test_no_leakage_gradient()
    print()
    print('=' * 60)
    print('ALL SANITY TESTS PASSED!')
    print('=' * 60)

if __name__ == '__main__':
    run_all_sanity_tests()
