# Repository Audit - HWIN-Net

**Date:** 2026-07-16  
**Workspace:** C:\\Users\\lenovo\\hwin_net  
**Current State:** Research implementation workspace with significant technical debt

---

## 1. Duplicate / Backup Files

### Model Files (Critical - contains scientific code, DO NOT MODIFY logic)
| File | Status | Action |
|------|--------|--------|
| models/schema_encoder.py.backup | Backup of schema_encoder.py | DELETE |
| models/schema_encoder_b64.txt | Base64 encoded version | DELETE |
| models/identifiability_gate.py.backup2 | Backup of identifiability_gate.py | DELETE |
| models/query_head.b64 | Base64 encoded query_head | DELETE |
| models/hwin_net_content.py | Duplicate of hwin_net.py | DELETE |
| models/temp.py, models/temp2.py | Temporary files | DELETE |
| models/test_b64.py, models/test_write.py, models/test_write.txt | Test artifacts | DELETE |

### Training Files
| File | Status | Action |
|------|--------|--------|
| training/trainer_old.py | Old trainer version | DELETE |

### Dataset Files
| File | Status | Action |
|------|--------|--------|
| datasets/dataset.py.b64 | Base64 encoded | DELETE |
| datasets/dataset.py.bak | Backup | DELETE |
| datasets/dataset_backup.py | Backup | DELETE |
| datasets/dataset_new.py | Duplicate | DELETE |

### Utils Config Files
| File | Status | Action |
|------|--------|--------|
| utils/config_b64.txt | Base64 encoded | DELETE |
| utils/config_v2.py | Duplicate | DELETE |
| utils/c_b64.txt | Base64 encoded | DELETE |
| utils/fix_config.py, utils/fix_config2.py, utils/fix_config3.py | Fix scripts | DELETE |
| utils/fix_escape.py, utils/fix_escape2.py | Fix scripts | DELETE |
| utils/fix_literal.py | Fix script | DELETE |
| utils/fix_recovery.py | Fix script | DELETE |
| utils/fix_remaining.py | Fix script | DELETE |
| utils/gen.py, utils/gen_config.py | Generator scripts | DELETE |
| utils/encode_config.py | Encoder script | DELETE |
| utils/write_config.py, utils/write_final.py, utils/write_it.py, utils/write_script.py | Writer scripts | DELETE |

---

## 2. Debug / Test / Temporary Scripts (Root)

| File | Status | Action |
|------|--------|--------|
| build_m6.py | Build script | DELETE |
| gen_m6.py | Generator script | DELETE |
| debug_train.py through debug_train6.py | Debug scripts | DELETE |
| fix_gate.py, fix_pca.py, fix_plateau.py, fix_pred_loss.py | Fix scripts | DELETE |
| decode_b64.py, decode_ds.py | Decoder scripts | DELETE |
| experix.py, patch_trainer.py | Experimental scripts | DELETE |
| simple_write.py, write_*.py (7 files) | Writer scripts | DELETE |
| test.py, test_b64.py, test_py.py, test_file.py, test_train.py, test_data_load.py | Test scripts | MOVE to tests/ or DELETE |
| test.b64, test.pkl, test.txt, test_out.txt, test123.txt, test_write.txt | Test artifacts | DELETE |

---

## 3. Generated / Output Directories

| Path | Type | Size | Action |
|------|------|------|--------|
| outputs/ | Training outputs | Multiple runs | EXCLUDE via .gitignore |
| outputs_* | Training outputs | Multiple variants | EXCLUDE via .gitignore |
| experiments/results/ | Experiment results | JSON + logs | EXCLUDE via .gitignore |
| checkpoints/ | Model checkpoints | .pt files | EXCLUDE via .gitignore |
| catboost_info/ | CatBoost training info | Logs | EXCLUDE via .gitignore |
| scripts/experiments/catboost_info/ | CatBoost info | Logs | EXCLUDE via .gitignore |
| hwin_net.egg-info/ | Package build artifact | | EXCLUDE via .gitignore |
| .pytest_cache/ | Pytest cache | | EXCLUDE via .gitignore |
| __pycache__/ (many) | Python bytecode | | EXCLUDE via .gitignore |

---

## 4. Data Files (Large - must exclude from repo)

| File | Size | Type |
|------|------|------|
| data/train.parquet | ~68 MB | Training data |
| data/val.parquet | ~22 MB | Validation data |
| data/test.parquet | ~23 MB | Test data |
| data/pivoted_groups.parquet | ~124 MB | Preprocessed data |
| data/stats/*.pt | Statistics files | Precomputed stats |
| data/station_folds_seed_*.json | ~2.2 MB each | Cross-validation folds |
| data/small/*.parquet | Small subsets | For testing |

**Action:** ALL data files must be excluded via .gitignore and NOT committed.

---

## 5. Config Variants (Keep core, consider consolidating)

| File | Purpose | Keep? |
|------|---------|-------|
| configs/config.yaml | Main frozen config | YES |
| configs/config_small.yaml | Small data test | YES (for dev) |
| configs/config_small_log1p.yaml | Small + log1p transform | YES (for dev) |
| configs/config_small_robust.yaml | Small + robustness | YES (for dev) |
| configs/config_test.yaml | Test config | YES (for dev) |
| configs/hydra_config.yaml | Hydra integration | YES |

---

## 6. Experiment Framework (Keep but clean)

| Component | Location | Keep? |
|-----------|----------|-------|
| Experiment registry | experiments/experiments_all.json | YES |
| Experiment configs | experiments/experiments_*.json | YES |
| Experiment scripts | scripts/experiments/run_*.py | YES |
| Results timestamps | experiments/results/*_20260714_* | NO - versioned results |

---

## 7. Publication Artifacts (Keep)

| File | Purpose |
|------|---------|
| Benchmark_Report.md | Full campaign documentation |
| Registered_Experiment.json / .md | Frozen protocol |
| SCIENTIFIC_VALIDATION_REPORT.json | Results summary |
| FINAL_VALIDATION_REPORT.md | Final report |
| REPRODUCIBILITY_MANIFEST.md / .json | Reproducibility |
| Root_Cause_Analysis.md, Scientific_Recovery_Report.md | Debug history |
| publication_tables/*.json | Publication tables |
| reviewer_reports/*.md | Peer review |

---

## 8. Test Files (Consolidate)

| File | Status | Action |
|------|--------|--------|
| tests/test_sanity.py | Sanity checks | KEEP - move to tests/ |
| tests/test_theory.py | Theory tests | KEEP - move to tests/ |
| tests/test_full_pipeline.py | Full pipeline test | KEEP - move to tests/ |

---

## 9. Scripts to Organize

| File | Category |
|------|----------|
| scripts/compute_stats.py | Data preprocessing |
| scripts/preprocess_data.py | Data preprocessing |
| scripts/experiments/run_*.py | Experiment runners |
| scripts/experiments/generate_figures.py | Figure generation |
| scripts/write_hwin.py, scripts/test_write.py, scripts/testz_writez.py, scripts/tz_writez.py | Test/writer scripts - DELETE |

---

## 10. Package Structure Issues

### Current Issues:
1. **No single package root** - modules scattered at root level
2. **pyproject.toml includes too many patterns** - include = [hwin_net*, models*, utils*, ...] 
3. **No standard src/ layout** - should use src/hwin_net or similar
4. **Missing entry points** - no CLI defined
5. **No requirements-dev.txt** - dev dependencies not separated

### Recommended Structure:
`
hwin-net/
src/hwin_net/           # Main package
  models/
  training/
  losses/
  inference/
  datasets/
  utils/
configs/                # YAML configs (keep)
data/                   # Data directory (gitignored)
scripts/                # CLI tools
examples/               # Example notebooks/scripts
tests/                  # Unit/integration tests
docs/                   # Documentation
.github/                # CI/CD
pyproject.toml          # Package metadata
requirements.txt        # Runtime deps
requirements-dev.txt    # Dev deps
README.md
LICENSE
CITATION.cff
CHANGELOG.md
CONTRIBUTING.md
CODE_OF_CONDUCT.md
SECURITY.md
.gitignore
`

---

## 11. Secret Audit Areas to Check

- No .env files found
- No obvious API keys in code (visual scan)
- Check all configs for credentials
- Check any .json files for tokens

---

## Summary: Files to DELETE (Safe - not scientific logic)

**Root temporary scripts:** ~40 files
**Backup files:** ~15 files  
**Debug/fix scripts:** ~20 files
**Test artifacts:** ~15 files
**Generated outputs:** Entire directories (outputs/, checkpoints/, catboost_info/, .pytest_cache/, hwin_net.egg-info/)
**Bytecode caches:** All __pycache__/

**Estimated cleanup:** ~100+ files removable, reducing repo size significantly.

---

## Next Steps (Phase 2-6)

1. **Phase 2:** Create clean repository structure
2. **Phase 3:** Fix pyproject.toml with proper package layout  
3. **Phase 4:** Write professional README.md
4. **Phase 5:** Secret audit
5. **Phase 6:** Comprehensive .gitignore
