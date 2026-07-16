# Editor Summary -- HWIN-Net Phase XXIII Benchmark Campaign

## Manuscript Overview
**Title**: HWIN-Net: A Theory-Grounded Schema-Aware Neural Architecture for Cross-Station Environmental Prediction
**Phase**: XXIII - Registered Benchmark Campaign
**Submission Type**: Standard (not Registered Report)
**Target**: Nature Machine Intelligence

---

## Editor Assessment

### Strengths
1. **Methodological Rigor**: Exemplary experimental registration with full pre-specification of 56 experiments, hypotheses, success/failure criteria, and statistical analysis plan. All configuration hashes frozen before any results (except classical baselines).
2. **Theoretical Foundation**: The HWIN-Net architecture derives from a re-axiomatized SIS (Scientific Inference System) with 6 modules (M1-M6) tracing to specific axioms/theorems. Theory-to-experiment traceability is maintained throughout.
3. **Reproducibility**: Complete hash registry (config, model, training, losses, utils, datasets, data, stats, station folds). Environment documented (Python 3.13, PyTorch 2.13+cpu, Windows 11, 8 cores, no CUDA).
4. **Ablation Architecture**: All 7 ablation configurations verified to instantiate correctly with expected parameter counts (full 383K, minimal 19K). Config key mappings validated.
5. **Classical Baselines**: Properly executed (10 models x 5 seeds). ExtraTrees achieves R2=0.401 +/- 0.178.

### Critical Gaps
1. **Main Result Missing**: BENCH-001 (HWIN-Net on HWIN-Bench) has 0/5 seeds complete. Only seed 42 running (epoch 3 of 100).
2. **Modern Baselines Missing**: BENCH-003 (7 tabular DL models), BENCH-004 (5 foundation/DG models), BENCH-005 (4 uncertainty methods) all unrun.
3. **Ablation Performance Unknown**: ABLA-001 through ABLA-007 only architecture-validated.
4. **Theory Predictions Untested**: THEO-001 through THEO-007 all pre-registered, zero executed.
5. **Numerical Instability**: Val_loss explosion observed at epoch 3 (30B vs ~60 prior). Unexplained.
6. **CPU Timeline Infeasible**: ~5000 hours for BENCH-001 alone on available hardware.

---

## Reviewer Consensus
- 6/6 reviewers: REJECT
- Primary reasons: (1) No main results, (2) Incomplete baselines, (3) CPU-only infeasible, (4) Numerical instability, (5) Theory unvalidated, (6) Ablations unrun

---

## Editorial Decision

**DECISION: REJECT**

This manuscript cannot be accepted in its current state. The experimental design is among the most rigorous I have seen for an ML submission -- but a design is not a paper. The core claim (HWIN-Net outperforms baselines on cross-station environmental prediction) has zero supporting evidence.

## Recommended Path Forward

### Option 1: Registered Report (Recommended)
Submit the *protocol* (this experimental design) for In-Principle Acceptance at Nature Machine Intelligence or another Registered Report venue. Then execute the campaign on GPU compute and submit the completed study. This protects the design rigor while acknowledging the compute reality.

### Option 2: Complete on GPU, Resubmit as New
Secure GPU cluster (est. 250 GPU-hours on 4-8 GPU node). Complete all 56 experiments. Resubmit as a new standard submission. Given the theoretical novelty and rigorous design, a completed campaign would be competitive for:
- Nature Machine Intelligence (if results strongly support theory)
- ICML / NeurIPS / ICLR (strong ML venue fit)
- JMLR (if emphasis on theoretical ML)

---

## Requirements for Any Future Submission

1. **BENCH-001 complete**: 5 seeds x 100 epochs with learning curves, final metrics + 95% CIs
2. **BENCH-002/003/004/005 complete**: All baselines on identical splits/seeds/preprocessing
3. **ABLA-001..007 complete**: Full training with performance, calibration, runtime
4. **THEO-001..007 complete**: All 7 predictions tested with PASS/FAIL/PARTIAL + evidence
5. **ROBU-001..006 complete**: Stress test results
6. **CALI-001..007 complete**: Full calibration analysis
7. **STAT-001..005 complete**: Statistical rigor on complete statistical analysis
8. **Numerical stability documented**: Loss component analysis, gradient norms, gate statistics
9. **Negative results reported**: Honest reporting of any FAIL outcomes

---

## Timeline Estimate
- GPU procurement: 2-4 weeks
- BENCH-001: ~1 week (5 seeds parallel)
- Baselines: ~1 week
- Ablations: ~2 weeks (7 configs x 5 seeds, can parallelize)
- Theory/Robustness/Calibration: ~1 week
- Analysis/Writeup: ~1 week
- **Total: ~6-8 weeks on 4-8 GPU cluster**

---

## Final Note
The authors have done exceptional work on the *design* side. The failure is purely computational. I would be happy to consider a completed version of this campaign. The theory is novel, the architecture is well-grounded, and the experimental design is publication-ready. Execution is the only barrier.

**Recommendation**: Pursue Registered Report route or secure GPU compute for full execution before any standard submission.
