# Minimal Repair List — HWIN-Net Phase XXII

Phase: Task 3 - Construct Minimal Repair List
Date: 2026-07-15
Status: Implementation Pending
Based on: Root_Cause_Analysis.md (Tasks 1-2 complete)

---

## Summary

Seven critical implementation bugs identified in Root Cause Analysis (Category C - Implementation Bugs). All map to theory-specification violations. Zero theory invalidation (A) or architecture mismatch (B) confirmed.

| Repair ID | Bug | Module | Theory Violation | Severity | Est. Effort |
|-----------|-----|--------|------------------|----------|-------------|
| R1 | Gate routing broadcasting [B,B] | M5 IdentifiabilityGate | T4, CC2 | Blocker | 5 min |
| R2 | NoLeakage loss sign inverted | M6 NoLeakageRegularizer | A5, CC5, C1 | Blocker | 5 min |
| R3 | hard_gate=False default config | Config | CC2 | Blocker | 1 min |
| R4 | r0 regressor collapse (Softplus + unnormalized O_card) | M5 IdentifiabilityGate | A4, T4 | Blocker | 30 min |
| R5 | Equivariance loss ~12,853 at init (random orthogonal) | M2 RecoveryModule | CC3 | Blocker | 20 min |
| R6 | M3 identity PCA basis (trivial projection) | M3 ManifoldRetraction | T1, CC4 | High | 40 min |
| R7 | Missing PCA basis update callback hook | Trainer | CC4 | Medium | Done* |

*R7 partially exists in trainer.py (PCABasisCallback) but needs activation + proper components count.

---

## Repair Details

### R1: Gate Routing Broadcasting Bug [BLOCKER]
File: models/identifiability_gate.py line ~116 in route()
Bug: sigma2_total = sigma2_aleat + sigma2_nonid where sigma2_nonid is [B,B] not [B] due to missing unsqueeze(-1) on gate_vals
Fix: sigma2_nonid = (1.0 - gate_vals).unsqueeze(-1) * sigma2_prior
Validation: sigma2_total.shape == (B,) not (B,B)

---

### R2: NoLeakage Loss Sign Inversion [BLOCKER]
File: models/no_leakage.py line 208 (in _adversarial_loss)
Bug: Current code actually RETURNS positive noleak_loss = lambda_mi * disc_loss for total loss - but Root Cause Analysis claims sign bug.
Action: VERIFY current code. If noleak_loss is positive in return dict, bug already fixed. If negative, flip sign.
Check: Unit test expects noleak_loss >= 0 - verify it passes.

---

### R3: Hard Gate Disabled by Default [BLOCKER]
File: configs/config.yaml line ~65
Current: hard_gate: true  <- Already True!
But: gate.r0_method: "lookup" with r0_init: 4.0 - lookup table ignores r0_init.
Fix: Change r0_method: "regressor" and r0_init: 3.0 (match data mean |O|=3.61)
Also: In GateConfig dataclass defaults: ensure hard_gate=True

---

### R4: r0 Regressor Collapse [BLOCKER]
File: models/identifiability_gate.py lines 45-60 (__init__) and compute_r0()
Root Causes:
1. Input O_card (30-60) not normalized - dominates one-hot platform (0/1)
2. nn.GELU() then nn.Linear(64,1) with Xavier init - pre-Softplus ~500 - Softplus(500)=500
   BUT actual output 0.08-0.72 - first layer output NEAR ZERO
3. Softplus() squashes negative pre-activations to ~0
4. Bias r0_init=4.0 on 64->1 layer, but ReLU/GELU kills gradient when pre-act negative

Fix (minimal):
```python
# Normalize O_card to [0,1] range
n_obs_norm = (n_obs / 100.0).unsqueeze(1)  # n_vars = 100

# r0_regressor: remove Softplus, use linear output with bias init
self.r0_regressor = nn.Sequential(
    nn.Linear(config.num_platforms + 1, config.r0_regressor_hidden),
    nn.GELU(),
    nn.Linear(config.r0_regressor_hidden, 1, bias=True)
)
with torch.no_grad():
    self.r0_regressor[-1].bias.fill_(config.r0_init)  # r0_init=3.0
```
Remove F.softplus(r0_vals) in compute_r0() - return raw linear output.

---

### R5: Equivariance Loss Explosion at Init [BLOCKER]
File: models/recovery_module.py (intertwiner init) + training/trainer.py (warmup)
Root Cause: Intertwiners initialized as random orthogonal matrices - ||R - I||_F^2 = d*k = 64*32 = 2048 per platform x 4 platforms = 8000. Weight 0.1 = 800 loss.
Fix:
1. Initialize intertwiners to Identity (not random orthogonal)
2. Keep equivariance warmup (already in trainer: 10 epochs)
3. Verify warmup factor applied to equiv_loss in loss computation

Implementation in recovery_module.py:
```python
# In __init__ of EquivariantMLP / RecoveryModule:
for a in range(num_platforms):
    if a == base_platform:
        self.intertwiners[a] = nn.Identity()  # Already Identity
    else:
        # Initialize to Identity, not random orthogonal
        W = torch.eye(latent_dim, latent_dim) + 0.01 * torch.randn(latent_dim, latent_dim)
        self.intertwiners[a].weight.data.copy_(W)
```

---

### R6: M3 Identity PCA Basis [HIGH]
File: models/manifold_retraction.py PCARetraction.__init__ + trainer.py PCABasisCallback
Root Cause: Default basis = torch.eye(latent_dim, pca_components) - projects to first k dims only. Trivializes manifold retraction (T1/CC4 violated).
Fix:
1. Change PCARetraction.__init__ to NOT default to identity - require basis or initialize randomly
2. Ensure PCABasisCallback in trainer properly updates basis every N epochs
3. Fix callback: pca_components should be config value (32), not batch_size (32)

Implementation:
```python
# In PCARetraction.__init__: remove default basis, raise if None
if basis is None:
    # Initialize with random orthogonal (will be updated by callback)
    basis = torch.randn(latent_dim, pca_components)
    basis, _ = torch.linalg.qr(basis)
    self.register_buffer("basis", basis)
```

---

### R7: PCA Basis Callback Activation [MEDIUM]
File: training/trainer.py lines 38-55 (PCABasisCallback) + __init__ line ~130
Issues:
1. pca_components=getattr(self.training_config, 'batch_size', 32) - should be retraction.pca_components (32)
2. Callback on_batch_end collects mu_final but needs eval() mode for consistency
3. update_basis called every 5 epochs - OK but verify its called

Fix: Pass correct pca_components from config, ensure callback active.

---

## Config Changes Required (configs/config.yaml)

```yaml
gate:
  r0_method: "regressor"      # was "lookup"
  r0_init: 3.0                # was 4.0 (match data mean |O|=3.61)
  hard_gate: true             # already true
  r0_regressor_hidden: 64

retraction:
  retraction_type: "pca"
  pca_components: 32
  # manifold_basis_path: null  # let callback compute

loss:
  lambda_equiv: 0.1           # keep, warmup handles scale
  equiv_loss_type: "frobenius"
```

---

## Verification Gates (Must Pass Before Benchmark)

| Test | Expected | File |
|------|----------|------|
| test_sanity_linear_regression | R2 > 0.9 | tests/test_sanity.py (new) |
| test_sanity_identity | R2 > 0.9 | tests/test_sanity.py (new) |
| test_gate_gradient_flow | Gate params have grad | tests/test_gate.py |
| test_noleak_sign | noleak_loss >= 0 | existing test |
| test_equivariance_warmup | Loss stable epoch 0-10 | trainer log |

---

## Execution Order

1. R1, R2, R3 - Config + 2 line fixes (10 min)
2. R4 - r0 regressor rewrite (30 min)
3. R5 - Intertwiner init + verify warmup (20 min)
4. R6 - PCA basis init + callback fix (40 min)
5. R7 - Callback activation (5 min)
6. Synthetic Sanity Tests - Run linear/identity/noise-free (60 min)

Total: ~3 hours

---

## Post-Repair: GO/NO-GO Criteria

| Criterion | Threshold |
|-----------|-----------|
| Synthetic linear regression R2 | > 0.9 |
| Synthetic identity R2 | > 0.9 |
| Gate gradient flow | All M5 params have non-zero grad |
| NoLeakage loss sign | noleak_loss >= 0 in total loss |
| Equivariance loss epoch 0 | < 100 (not 12,853) |
| M3 idempotence loss | > 0 initially (not 0 from identity) |
| Training stable 10 epochs | No NaN, loss decreases |

If ALL pass - Proceed to HWIN-Bench integration (Tasks 6-12).
If ANY fail - STOP, debug, repeat.
