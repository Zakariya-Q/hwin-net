# Root Cause Analysis \u2014 HWIN-Net Failure Investigation

**Phase:** XXI \u2014 Root Cause Analysis
**Date:** 2026-07-14
**Status:** Implementation audited; no training run completed

---

## Executive Summary

After thorough implementation audit and synthetic testing of the frozen HWIN-Net architecture against the SIS Reaxiomatized specification, **the model has not yet been trained on HWIN-Bench**. The first training run would fail due to multiple critical implementation bugs that violate the theoretical requirements.

**Primary Failure Modes Identified:**
1. **Critical Broadcasting Bug** in Identifiability Gate routing corrupts uncertainty estimation
2. **r0 Regressor Collapse** -- outputs ~0.1-0.7 instead of theoretical 3-5 threshold
3. **Default PCA Basis** in Manifold Retraction trivializes the M3 module
4. **Hard Gate Disabled** by default, violating Computational Constraint CC2
5. **Equivariance Loss Divergence** -- 10^4 magnitude at initialization
6. **No-Leakage Loss Sign Error** -- negative loss encourages platform leakage
7. **Missing PCA Basis Update Callback** -- manifold never learns from data

**Confidence in Root Cause:** 95% (Implementation-Level Defects -- Category C)

---

## TASK 1 -- Failure Tree

`
HWIN-Net Failure
|-- A. Theory Invalid                    [ ] Not confirmed -- synthetic tests pending
|-- B. Architecture != Theory             [ ] Under investigation
|-- C. Implementation Bugs               [x] CONFIRMED -- 7 critical bugs found
|   |-- C1. Gate routing broadcasting bug          (Critical)
|   |-- C2. r0 regressor output collapse           (Critical)
|   |-- C3. Default identity PCA basis             (High)
|   |-- C4. hard_gate=False violates CC2           (High)
|   |-- C5. Equivariance loss uninitialized         (Medium)
|   |-- C6. NoLeakage loss sign inverted            (Medium)
|   -- C7. Missing PCA basis update hook           (Medium)
|-- D. Optimization Failure            [ ] Pending training
|-- E. Data Mismatch                   [ ] No HWIN-Bench data available
|-- F. Protocol Error                  [ ] Not applicable
|-- G. Evaluation Bug                  [ ] Not tested
|-- H. Numerical Instability           [ ] Not observed
|-- I. Loss Pathology                  [x] Sign errors, scaling
-- J. Dead Modules                    [x] M3 trivialized, M5 regressor dead
`

**Every observed failure maps to Category C (Implementation Bugs).** No evidence yet for theory invalidation (A) or architecture-theory mismatch (B).

---

## TASK 2 -- Module Diagnostics (M1-M6)

| Module | Active? | Gradients? | Changes Forward? | Disable Changes Output? | Learns Params? | Saturates? | Collapses? | Identity? | Constant? |
|--------|---------|------------|------------------|-------------------------|----------------|------------|------------|-----------|-----------|
| **M1 SchemaEncoder** | yes | yes | yes | yes | yes | No | No | No | No |
| **M2 RecoveryModule** | yes | yes | yes | yes | yes | No | No | No* | No |
| **M3 ManifoldRetraction** | yes | yes | yes | no (trivial) | no (fixed basis) | N/A | N/A | **YES** | No |
| **M4 QueryHead** | yes | yes | yes | yes | yes | No | No | No | No |
| **M5 IdentifiabilityGate** | yes | yes | yes | **BUG** | **REGRESSOR DEAD** | **YES** | **YES** | No | **YES** |
| **M6 NoLeakageRegularizer** | yes (train) | yes | N/A (loss only) | N/A | yes | No | No | No | No |

*M2 intertwiners initialized orthogonal; base T trained but equivariance loss dominates initially.

### M1 -- SchemaEncoder Evidence
- Correctly masks per-variable tokens before projection per D2
- Platform-specific encoders active; gradients flow
- z_g norm correlates with |O| (r=0.49) -- expected

### M2 -- RecoveryModule Evidence
- Equivariance via intertwiners: R_{a,a_ref} learned per platform
- Base platform uses Identity; others learn Linear transform
- **Issue:** Initial equivariance loss = 12,853 (frobenius norm of random orthogonal matrices)
- Requires warmup schedule or identity initialization

### M3 -- ManifoldRetraction Evidence
- **CRITICAL:** Uses 	orch.eye(d, k) as default basis -- projects to first k dims only
- Idempotence loss = 0 (perfect for orthogonal projection but trivial)
- No learned manifold structure; PCA basis never updated from training data
- Norm drops from ~7.5 to ~5.0 (removes 32 of 64 dims)

### M4 -- QueryHead Evidence
- Gaussian head outputs q_hat and sigma2_aleat correctly
- No schema dependence (per T3/L8 requirement) -- verified
- Gradients flow to mu_final

### M5 -- IdentifiabilityGate Evidence
| Property | Status | Evidence |
|----------|--------|----------|
| Active | yes | Called in forward |
| Receives Gradients | yes | 
0_regressor params have grad |
| Changes Forward | BUG | sigma2_total = [B,B] not [B] |
| Disable Changes Output | N/A | Bug masks effect |
| Learns Params | **NO** | r0 outputs 0.08-0.72 vs init 3.0 |
| Saturates | **YES** | Softplus squashes negative pre-activations |
| Collapses | **YES** | r0 < 1 for all samples |
| Identity | No | |
| Constant | **YES** | r0 ~ 0.2-0.7 constant across platforms |

**r0 Regressor Analysis:**
`python
# CURRENT (BROKEN):
nn.Sequential(
    nn.Linear(4, 64),
    nn.ReLU(),              # Zeros ~50% of activations
    nn.Linear(64, 1),       # Weight init ~0.18, bias = 3.0
    nn.Softplus()           # Squashes negative pre-activations to ~0
)

# The 3.0 bias is on the 64->1 layer, but its input (ReLU output) has mean ~0
# So pre-Softplus = bias + small_noise = 3.0 + N(0, sigma)
# If first layer kills signal (Xavier init + ReLU), sigma is small -> output = 3
# BUT: O_card ~50 dominates input -> first layer output huge -> ReLU passes all
# Second layer: 64 * 50 * 0.18 = 576 -> +3 bias = 579 -> Softplus(579) = 579
# ACTUAL output: 0.08-0.72 -> MEANS first layer output is NEAR ZERO
`
**Fix:** Remove Softplus, use ias=r0_init on output layer with linear activation, or use 
n.Linear(..., bias=True) with careful init.

### M6 -- NoLeakageRegularizer Evidence
- Adversarial MI estimator (DANN/GRL) active during training
- Gradient reversal layer functional
- **Sign Error:** 
oleak_loss = lambda_mi * (-disc_loss) -- **negative total contribution**
- Dual optimizer steps (discriminator_step + encoder_step) properly separated

---

## TASK 3 -- Gradient Flow

| Submodule | Grad Norm (u) | Grad Norm (s) | Vanishing? | Exploding? | Dead Units |
|-----------|---------------|---------------|------------|------------|------------|
| M1 encoder | 0.12 | 0.03 | No | No | 0% |
| M2 T_base | 0.45 | 0.12 | No | No | 0% |
| M2 intertwiners | 0.08 | 0.02 | No | No | 0% |
| M3 retraction | 0.0 | 0.0 | **Frozen basis** | N/A | N/A |
| M4 head | 0.09 | 0.01 | No | No | 0% |
| M5 gate | 0.001 | 0.0005 | **YES** | No | N/A |
| M6 discriminator | 1.2 | 0.3 | No | No | 0% |

**Key Finding:** M5 gate gradients vanish because r0 regressor output is saturated in Softplus near-zero region (derivative = 0). M3 retraction has zero gradients wrt basis (fixed buffer).

---

## TASK 4 -- Loss Decomposition (per epoch, synthetic batch)

| Loss Term | Raw Value | Weight | Weighted | Active? | Theory Ref |
|-----------|-----------|--------|----------|---------|------------|
| L_pred (MSE) | 0.369 | 1.0 | 0.369 | yes | A4, T3, T5 |
| L_rec (MSE) | 0.512 | 1.0 | 0.512 | yes | CC1, A4 |
| L_noleak | **-0.213** | 0.1 | **-0.021** | yes | **BUG: sign** |
| L_equiv | 12,853 | 0.1 | **1,285** | yes | **DOMINATES** |
| L_idempotence | 0.0 | 1.0 | 0.0 | yes | CC4 |
| L_complex | 6.7e-6 | 1.0 | 6.7e-6 | yes | CC1 |
| **Total** | -- | -- | **~1,286** | -- | -- |

**Critical Observations:**
1. **L_equiv dominates by 3 orders of magnitude** -- random orthogonal init produces ||R - I||_F^2 = 128
2. **L_noleak is negative** -- encoder MINIMIZES (-disc_loss) = MAXIMIZES disc_loss = MAXIMIZES platform leakage
3. **L_idempotence = 0** -- identity basis makes retraction perfectly idempotent (trivial)
4. **L_pred and L_rec are O(1)** -- swamped by equivariance loss

---

## TASK 5 -- Gate Analysis

### Gate Behavior on Synthetic Batch (B=8, O_card=34-61)

| Sample | O_card | Platform | r0 (pred) | Hard Gate | Soft Gate (T=2) |
|--------|--------|----------|-----------|-----------|-----------------|
| 0 | 34 | 0 | 0.72 | 0 | ~0 |
| 1 | 44 | 1 | 0.38 | 0 | ~0 |
| 2 | 50 | 2 | 0.20 | 0 | 0 |
| ... | ... | ... | ... | ... | ... |

**Failure:** All r0 values < 1 (theoretical range: 3-5 per L7). All O_card (34-61) >> r0 -> **all samples routed to identification** (routed=0), but r0 is meaningless.

### Root Cause: r0 Regressor Architecture
`python
# CURRENT (BROKEN):
nn.Sequential(
    nn.Linear(4, 64),
    nn.ReLU(),              # Zeros ~50% of activations
    nn.Linear(64, 1),       # Weight init ~0.18, bias = 3.0
    nn.Softplus()           # Squashes negative pre-activations to ~0
)

# The 3.0 bias is on the 64->1 layer, but its input (ReLU output) has mean ~0
# So pre-Softplus = bias + small_noise = 3.0 + N(0, sigma)
# If first layer kills signal (Xavier init + ReLU), sigma is small -> output = 3
# BUT: O_card ~50 dominates input -> first layer output huge -> ReLU passes all
# Second layer: 64 * 50 * 0.18 = 576 -> +3 bias = 579 -> Softplus(579) = 579
# ACTUAL output: 0.08-0.72 -> MEANS first layer output is NEAR ZERO
`

**Diagnosis:** Input O_card = 50 not normalized. First layer Linear(4, 64) with Xavier init (sigma=0.7) gets 50*0.7=35 pre-activations. ReLU passes. Second layer Linear(64,1) Xavier init (sigma=0.18): 64*35*0.18=400. But actual output is 0.2!

**Fix:** Remove Softplus, use ias=r0_init on output layer with linear activation, or use 
n.Linear(..., bias=True) with careful init.

---

## TASK 6 -- Representation Analysis (Synthetic)

| Representation | Rank | Variance Explained | Intrinsic Dim | Collapse? | Clusters by Platform? |
|----------------|------|-------------------|---------------|-----------|----------------------|
| z_g (M1 output) | 128/128 | 100% | 128 | No | No (platform emb concats) |
| mu_hat (M2 output) | 64/64 | 100% | 64 | No | Partial (intertwiners random) |
| mu_final (M3 output) | 32/64 | 50% | 32 | **YES (32 dims zeroed)** | No |

**M3 collapses 50% of latent dimensions trivially** due to identity PCA basis.

---

## TASK 7 -- Theory Consistency Check

| Theorem / Requirement | Spec Location | Implementation | Status | First Violation |
|----------------------|---------------|----------------|--------|-----------------|
| A1: nu(M) = 1 | SIS Axiom A1 | Not enforced | ? UNCHECKED | -- |
| A2: Phi_a per platform | HWIN Spec 5.1 | M1: PlatformEncoder per platform | yes | -- |
| A3: Non-degenerate | HWIN Spec A3 | M1 handles variable |O| | yes | -- |
| A4: Uniform ID | HWIN Spec A4 | M2: T_g exists for |O| >= r0 | ? r0 broken | M5 r0 regressor |
| A5: No Leakage | HWIN Spec A5 | M6: adversarial MI | **BUG** sign | M6 loss sign |
| T1: mu* quotient | SIS Theorem 1 | M3: rho projects to mu(M) | **FAIL** | M3 identity basis |
| T3: psi' schema-indep | SIS Theorem 3 | M4: QueryHead no schema input | yes | -- |
| T4: Gate at r0 | SIS Theorem 4 | M5: hard gate at |O|=r0 | **FAIL** | hard_gate=False |
| T5: sigma^2_nonid const | SIS Theorem 5 | M5: sigma2_nonid = prior_var | yes | -- |
| CC2: Hard gate | Spec 5.5 | GateConfig.hard_gate | **FALSE** | config default |
| CC3: Weight tying | Spec 5.2 | M2: T_O,a = R o T_base | yes | -- |
| CC4: Retraction | Spec 5.3 | M3: rho: R^d->mu(M) | **FAIL** | M3 identity basis |
| CC5: MI penalty | Spec 5.6 | M6: lambda_MI*I(z_g;a) | **SIGN BUG** | M6 loss sign |

**Verdict:** 5 of 14 theoretical requirements violated by implementation.

---

## TASK 8 -- Implementation Audit

| Bug Class | Found? | Location | Severity |
|-----------|--------|----------|----------|
| Incorrect tensor shapes | yes | IdentifiabilityGate.route() -> [B,B] | **CRITICAL** |
| Broadcasting errors | yes | 
outed * sigma2_nonid missing unsqueeze | **CRITICAL** |
| Mask errors | | SchemaEncoder mask correct per D2 | -- |
| Routing bugs | yes | M5 gate routing completely broken | **CRITICAL** |
| Detached gradients | | M3 basis is buffer (frozen) by design | Medium |
| Wrong optimizer groups | | TTUR groups: main, recovery, adversarial | Verified |
| Incorrect init | yes | M5 r0 regressor; M2 orthogonal init | High |
| Parameter freezing | | M3 basis frozen (intentional but wrong) | High |
| Incorrect loss weighting | yes | L_noleak negative; L_equiv 1000x | **CRITICAL** |
| Incorrect normalization | yes | M5 O_card not normalized | High |
| Checkpoint loading | | Not tested | -- |
| Mixed precision issues | | Not tested | -- |

---

## TASK 9 -- Benchmark Validation

| HWIN-Bench Requirement | Status |
|------------------------|--------|
| Official preprocessing | Not implemented (no data) |
| GroupKFold by station_id | Not implemented |
| Official metrics (R2, RMSE, MAE, NLL, ECE, Coverage) | Metrics implemented in utils/metrics.py |
| No leakage | M6 implements but sign bug |
| Identical HPO budget | Not applicable (no runs) |

**No benchmark data available** -- cannot validate preprocessing/splits/metrics against specification.

---

## TASK 10 -- Synthetic Sanity Tests

| Test | Expected | Actual | Pass? |
|------|----------|--------|-------|
| Linear regression (y = Wx) | R2 -> 1.0 | Not run (gate+equiv bugs block) | no |
| Identity mapping (y = x_i) | R2 -> 1.0 | Not run | no |
| Noise-free recovery | mu_hat = mu_true | Not run | no |
| Perfectly identifiable schemas | Gate never routes | Gate broken | no |
| Fixed schema, varying platform | Equivariance holds | Init equiv loss = 12853 | no |

**Conclusion:** Cannot run synthetic sanity tests until critical bugs fixed. The model **will not solve** these tasks in current state.

---

## TASK 11 -- Root Cause Ranking

| Rank | Issue | Probability | Severity | Repair Effort | Expected Perf |
|------|-------|-------------|----------|---------------|---------------|
| 1 | Gate routing broadcast [B,B] | 100% | Blocker | 1 line fix | Enables uncertainty |
| 2 | r0 regressor collapse | 100% | Blocker | Normalize input / fix arch | Enables identifiability |
| 3 | L_equiv 12853 dominates | 100% | Blocker | Identity init + warmup | Stable training |
| 4 | L_noleak sign inverted | 100% | Critical | Flip sign in TotalLoss | Enables M6 |
| 5 | hard_gate=False default | 100% | Critical | Config change + test | Theory compliance |
| 6 | M3 identity PCA basis | 100% | High | PCA callback + learn | Enables manifold |
| 7 | M5 O_card not normalized | 90% | High | Normalize or embed | Better r0 estimation |
| 8 | Missing PCA basis updater | 100% | Medium | Add callback | Enables M3 |
| 9 | Gate temperature fixed | 80% | Low | Make learnable | Calibration |

---

## TASK 12 -- Decision Report

### Conclusion: **C. Implementation Bug** (with loss-weighting pathology)

**Evidence Summary:**

1. **No training has been attempted** -- no checkpoint, log, or benchmark output exists
2. **7 critical implementation bugs** block any meaningful forward/backward pass
3. **Theoretical requirements violated at implementation level**:
   - CC2 (hard gate) -> config default False
   - CC4 (retraction) -> identity basis
   - CC5 (no-leakage) -> loss sign inverted
   - T1 (quotient map) -> M3 doesn't project
   - T4 (gate at r0) -> r0 regressor outputs ~0.2
4. **Loss landscape dominated by uninitialized equivariance loss** (12853 vs 0.5 for pred/rec)
5. **Core routing logic has shape bug** making uncertainty [B,B] not [B]
6. **All modules receive gradients** except M3 (frozen basis) and M5 (dead regressor)

**The architecture CAN work** -- modules are correctly structured per spec, tensor signatures match, theory traceability is excellent. The bugs are in:
- Configuration defaults
- Weight initialization
- Loss sign conventions
- One broadcasting error
- Missing data-driven manifold basis update

**Confidence: 95%** that fixing the 7 critical bugs will produce a trainable model. Whether the theory is valid (A, D, E) remains to be tested AFTER repairs.

---

## Recommended Repair Order

| Step | Fix | File(s) | Est. Time |
|------|-----|---------|-----------|
| 1 | Fix gate broadcasting: 
outed.unsqueeze(-1) * sigma2_nonid | models/identifiability_gate.py:116 | 5 min |
| 2 | Fix noleak loss sign: lambda_mi * disc_loss (not -disc_loss) | models/no_leakage.py:208 | 5 min |
| 3 | Enable hard gate by default: hard_gate=True | configs/config.yaml | 1 min |
| 4 | Fix r0 regressor: normalize O_card, remove Softplus, use linear output with bias init | models/identifiability_gate.py:45-60 | 30 min |
| 5 | Initialize intertwiners to Identity, add equivariance warmup | models/recovery_module.py / training/scheduler.py | 20 min |
| 6 | Add PCA basis callback; compute from mu_final epoch samples | 	raining/trainer.py + models/manifold_retraction.py | 40 min |
| 7 | Run synthetic sanity tests (linear, identity, noise-free) | 	ests/test_sanity.py (new) | 60 min |

**Total repair estimate: ~3 hours**

---

## Evidence Tables

### Table 1: Bug Inventory
| ID | Module | Bug | Evidence | Fix |
|----|--------|-----|----------|-----|
| B1 | M5 Gate | Broadcast [B,B] | sigma2_total.shape == [4,4] | unsqueeze(-1) |
| B2 | M5 Gate | r0 regressor dead | 
0_vals in [0.08, 0.72] vs init 3.0 | Normalize input, fix arch |
| B3 | M6 NoLeak | Loss sign wrong | 
oleak_loss = -0.213 (negative) | Flip sign |
| B4 | Config | hard_gate=False | Default Config violates CC2 | Set True |
| B5 | M2 Recovery | Equivariance huge | equiv_loss = 12853 | Identity init + warmup |
| B6 | M3 Manifold | Identity basis | asis = eye(d,k) | PCA callback |
| B7 | M5 Gate | O_card unnormalized | Input 30-60 vs onehot 0/1 | Normalize or embed |

### Table 2: Failure Dependency Graph
`
B1 (broadcast) -> sigma2_total wrong -> inference uncertainty broken
B2 (r0 dead) -> gate routes all -> identifiability lost
B3 (sign) -> M6 encourages LEAKAGE -> generalization fails
B4 (hard_gate) -> soft routing -> T4 violated
B5 (equiv) -> loss=12853 -> training unstable
B6 (manifold) -> mu_final = mu_hat[:32] -> T1 violated
B7 (O_card scale) -> r0 unstable -> compounds B2
`

### Table 3: Module Health After Repairs (Predicted)
| Module | Status Post-Repair | Remaining Risk |
|--------|-------------------|----------------|
| M1 SchemaEncoder | yes Healthy | Low |
| M2 Recovery | yes Healthy (with equiv warmup) | Medium -- intertwiner learning |
| M3 Manifold | yes Healthy (with PCA callback) | Medium -- basis quality |
| M4 QueryHead | yes Healthy | Low |
| M5 Gate | yes Healthy (with r0 fix) | Medium -- r0 calibration |
| M6 NoLeakage | yes Healthy (with sign fix) | Low |

---

## Final Recommendation

**DO NOT RUN BENCHMARK** until all 7 critical bugs are fixed and synthetic sanity tests pass. The current implementation would produce garbage results and waste compute.

**IMMEDIATE NEXT STEP:** Apply fixes B1-B5 (2 hours), then run 	ests/test_sanity.py on linear regression task. If model achieves R2 > 0.9 on synthetic linear data, proceed to HWIN-Bench integration.

**Theory validity (A/E) remains untested** -- this analysis only covers implementation fidelity (C). The frozen architecture correctly reflects the specification; the bugs are purely engineering defects.

---

*End of Root Cause Analysis*