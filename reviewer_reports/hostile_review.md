# Hostile Review -- HWIN-Net Phase XXIII Benchmark Campaign

## Reviewer 1: Where are the results

Rating: REJECT

This submission presents an experimental design, not an empirical paper. The authors have registered 56 experiments, validated their ablation architecture, and run classical baselines -- but the main result (BENCH-001: HWIN-Net on HWIN-Bench) has only been run for 3 epochs on 1 of 5 seeds. The training loss curves show loss explosion at epoch 3 (val_loss 30B) in early runs. Zero main results for the proposed method.

## Reviewer 2: Incomplete baseline suite

Rating: REJECT

Only classical ML baselines (10 models) are complete. Modern tabular DL (FT-Transformer, SAINT, TabTransformer, TabNet, TabM, DeepSets, SetTransformer) are unrun. Foundation models (TabPFN) and domain generalization (IRM, GroupDRO, CORAL, MMD) are unrun. Uncertainty baselines (MC-Dropout, Deep Ensembles, SWAG, Conformal) are unrun. A Nature MI paper requires competitive baselines across all categories.

## Reviewer 3: CPU-only makes this infeasible

Rating: REJECT

Training on full HWIN-Bench (2.8M samples) takes ~10 hours/epoch on CPU. 5 seeds x 100 epochs = 5000 hours for BENCH-001 alone. Ablations add 7x more. The authors acknowledge this but have not secured GPU compute. This is a design document, not a completed study. Without GPU results, the paper cannot be evaluated.

## Reviewer 4: Numerical stability unexplained

Rating: REJECT

Early runs show val_loss exploding to 30 billion at epoch 3 (from ~60 at epoch 0-2). Gradient clipping (norm=1.0) is configured but the explosion suggests deeper issues: possible gradient flow through equivariance loss, adversarial training instability, or PCA retraction numerical issues. No ablation of loss components shown. This must be resolved before any results are credible.

## Reviewer 5: Theory validation entirely untested

Rating: REJECT

Seven theorem-derived predictions (THEO-001 through THEO-007) are pre-registered but zero have been tested. The identifiability gate, no-leakage regularizer, equivariant recovery, schema encoder, and manifold retraction all make strong claims with zero empirical evidence. A theory-grounded paper must validate its theory.

## Reviewer 6: Ablation only architectural

Rating: REJECT

The ablation study (7 configs) only verifies instantiation and parameter counts. No training completed, no performance metrics, no runtime comparisons. The minimal model (19K params) vs full (383K) is a 20x difference but no data on whether the extra parameters buy anything. Architectural validation is necessary but not sufficient.

## Editor Summary

The HWIN-Net campaign represents a methodologically exemplary experimental design -- registered, frozen, with clear hypotheses and pre-specified analysis. However, the computational execution is incomplete. The main benchmark (BENCH-001) has not been run to completion due to hardware constraints. The ablation studies, theory validations, and modern baseline comparisons are all architecture-validated but lack empirical results. As a Nature Machine Intelligence submission, this would be rejected without full empirical validation.

## Final Decision: REJECT (Category D -- Requires GPU Cluster Before Submission)

Resubmission Requirements:
1. Secure GPU cluster (A100/H100, 4-8 GPUs) -- reduces epoch time from 10h to ~15min
2. Run BENCH-001 (5 seeds, 100 epochs) -- est. 18 GPU-hours
3. Run BENCH-003, BENCH-004, BENCH-005 -- est. 40 GPU-hours
3. Run ablations (7 configs x 5 seeds x 100 epochs) -- est. 125 GPU-hours
4. Run theory validation + robustness + calibration -- est. 50 GPU-hours
5. Total estimate: ~250 GPU-hours on 4-8 GPU node
6. With results: Re-evaluate for Category A (Nature MI) or B (ICML/NeurIPS/ICLR)
