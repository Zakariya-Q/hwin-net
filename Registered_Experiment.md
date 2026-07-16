# Registered Experiment — HWIN-Net Phase XXIII

**Registration Date:** 2026-07-15T09:48:32.836633
**Phase:** XXIII - Registered Benchmark Campaign
**Status:** FROZEN — No modifications allowed after this point

---

## 1. Code Hashes (Immutable)

| Component | SHA256 (truncated) |
|-----------|-------------------|
| Config (configs/config.yaml) | b64fbec4d21b34d99b78f2d6f73a3cfcedd331c897573243eaeee2a32cbeb21d |
| Models (models/*.py) | ffb95ee64038c320 |
| Training (training/*.py) | 44f3a35d5f441a8a |
| Datasets (datasets/*.py) | bd5036cc5eefff6f |
| Losses (losses/*.py) | 3a12bcec9a35c174 |
| Utils (utils/*.py) | ba3a388d9270cbe3 |

---

## 2. Data Versions (Immutable)

| File | SHA256 |
|------|--------|
| train.parquet | 357b1dc7c3d76c677e36cbf60ac0dc24bab949211495959320175f8342b0a237 |
| val.parquet | da4530cbd9baabf9d05ddb2366cada946f47962a9f0e43f77ca561e76df6ff29 |
| test.parquet | 1cccedd07b7d46c9dcbe5d8b80f016bc72e4c06107c67fcbdaa1cba0f03bbf0a |

| File | SHA256 |
|------|--------|
| stats/feature_mean.pt | 7ff1762b4295ba0353ecd77fcf2052dab31d18b19208ec846b64195320be868b |
| stats/feature_std.pt | a8c15e4455f717afedf7c9a63e2949c1cd83d4496a07984267ea06adf70c0085 |
| stats/pca_basis.pt | 009f11016b02ab0a03982e9a185104a45b028f6334c798c3b6a02923eaade8e9 |
| stats/feature_columns.txt | f2c12797a38cd565804e5ddfce16eabb06646c2391916946814b4d94ed889e06 |


---

## 3. Random Seeds (Fixed)
- **Seed 42**
- **Seed 123**
- **Seed 456**
- **Seed 789**
- **Seed 999**


---

## 4. Hardware & Environment

| Property | Value |
|----------|-------|
| Device | CPU-only (no CUDA) |
| CPU | Intel/AMD, 8 cores |
| RAM | System dependent (min 32 GB recommended) |
| OS | Windows 11 10.0.26200 |

### Software Versions
| python | 3.13.3 |
| torch | 2.13.0+cpu |
| numpy | 2.3.2 |
| polars | 1.42.1 |
| pandas | 2.3.2 |
| sklearn | 1.7.2 |
| xgboost | 3.3.0 |
| lightgbm | 4.6.0 |
| catboost | 1.2.10 |
| platform | Windows-11-10.0.26200-SP0 |


---

## 5. Key Configuration (Frozen)

| Parameter | Value |
|-----------|-------|
| num_variables | 100 |
| num_platforms | 5 |
| encoder.output_dim | 128 |
| recovery.latent_dim | 64 |
| retraction.pca_components | 32 |
| gate.hard_gate | True |
| gate.r0_method | regressor |
| gate.r0_init | 2.5 |
| training.batch_size | 32 |
| training.val_batch_size | 64 |
| training.max_epochs | 100 |
| training.early_stopping_patience | 20 |
| lr_main | 0.001 |
| lr_recovery | 0.0005 |
| lr_adversarial | 0.0002 |
| loss.lambda_pred | 1.0 |
| loss.lambda_rec | 1.0 |
| loss.lambda_noleak | 0.1 |
| loss.lambda_equiv | 0.1 |
| loss.lambda_complex | 0.0001 |
| loss.lambda_idempotence | 1.0 |
| training.equivariance_warmup_epochs | 10 |


---

## 6. Benchmark Protocol (Fixed)

- **Dataset:** HWIN-Bench (official splits)
- **Preprocessing:** Official (feature normalization via training stats, GroupKFold by station_id)
- **Cross-validation:** 5-fold GroupKFold by station_id
- **Seeds:** 5 seeds (42, 123, 456, 789, 999)
- **No manual intervention** after training begins
- **Metrics:** R², RMSE, MAE, NLL, ECE, Coverage, 95% CI

---

## 7. Baselines (Fixed)

All baselines use identical splits, preprocessing, and hardware:

1. **Classical ML:** Linear, Ridge, Lasso, ElasticNet, RF, ExtraTrees, GB, XGBoost, LightGBM, CatBoost
2. **Modern Tabular DL:** FT-Transformer, SAINT, TabTransformer, TabNet, TabM, DeepSets, SetTransformer
3. **Foundation/DG:** TabPFN (if feasible), IRM, GroupDRO, CORAL, MMD
4. **Uncertainty:** MC-Dropout, Deep Ensembles, SWAG, Conformal Prediction

---

## 8. Ablation Plan (Fixed)

| ID | Module | Description |
|----|--------|-------------|
| ABLA-001 | M1 | Remove Schema Encoder |
| ABLA-002 | M2 | Remove Recovery Module |
| ABLA-003 | M3 | Remove Manifold Retraction |
| ABLA-004 | M4 | Remove Query Head |
| ABLA-005 | M5 | Remove Identifiability Gate |
| ABLA-006 | M6 | Remove No-Leakage Regularizer |
| ABLA-007 | MIN | Minimal (all modules disabled) |

---

## 9. Theory Validation Plan (Fixed)

| ID | Prediction | Test |
|----|------------|------|
| THEO-001 | Gate reduces negative transfer | Compare gate-on vs gate-off on unidentifiable schemas |
| THEO-002 | Refusal improves reliability | Calibrate refused predictions |
| THEO-003 | No-leakage improves OOD | Cross-station generalization |
| THEO-004 | Equivariant recovery enables transfer | Cross-platform performance |
| THEO-005 | Uncertainty decomposition identifies unidentifiable | sigma2_nonid correlation with |O| < r0 |

---

## 10. Robustness Plan (Fixed)

Evaluate: missing variables, noise injection, platform shift, unseen stations, unseen datasets, OOD

---

## 11. Calibration Plan (Fixed)

Compute: ECE, NLL, coverage, reliability diagrams, prediction intervals

---

## 12. Scientific Analysis Plan (Fixed)

Determine: variable transferability, schema failure modes, uncertainty patterns, gate activation patterns, transfer break points

---

## 13. Statistical Analysis Plan (Fixed)

Report: 95% CI, paired t-tests, effect sizes (Cohen's d), critical difference diagrams, Bonferroni correction

---

## 14. Reproducibility Verification (Fixed)

Re-run every experiment with same seed → verify exact numerical match

---

## 15. Final Decision Criteria (Fixed)

| Outcome | Criteria |
|---------|----------|
| A: Nature MI | Strong across all metrics, novel insight, no fatal flaws |
| B: ICML/NeurIPS/ICLR | Strong on core metrics, clear contribution, minor limitations |
| C: Environmental ML venue | Niche contribution, limited generality |
| D: Redesign required | Fails core benchmarks, theory invalidated, fatal flaws |

**Decision based ONLY on experimental evidence.**

---

*This registration is FINAL. No changes permitted after this point.*
