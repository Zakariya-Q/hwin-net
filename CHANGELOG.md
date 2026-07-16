# Changelog

All notable changes to HWIN-Net will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-07-16

### Added
- Initial release of HWIN-Net v0.1.0
- Six-module theory-derived architecture (M1-M6)
  - M1: SchemaEncoder with platform-specific encoding and masking
  - M2: RecoveryModule with equivariant weight tying
  - M3: ManifoldRetraction with PCA/VAE projection
  - M4: QueryHead with aleatoric/epistemic uncertainty
  - M5: IdentifiabilityGate with hard binary refusal
  - M6: NoLeakageRegularizer with adversarial MI estimation
- HWINNet integrated model
- HWIN-Bench dataset integration with GroupKFold by station_id
- TTUR optimizer with layer-wise learning rates
- Comprehensive loss suite (L_pred, L_rec, L_noleak, L_equiv, L_complex)
- Inference engine with uncertainty quantification
- Reproducible training with 5 seeds (42, 123, 456, 789, 999)
- Full test suite (26 tests passing)
- Configuration via YAML (Hydra-compatible)
- src-layout Python package (pip install -e .)

### Fixed
- Import paths for src-layout package
- pyproject.toml with proper setuptools configuration
- Package metadata and version management

### Documentation
- Comprehensive README.md
- LICENSE (MIT)
- CITATION.cff
- CONTRIBUTING.md
- CODE_OF_CONDUCT.md
- SECURITY.md
- CHANGELOG.md

### Infrastructure
- GitHub Actions CI workflow
- Lightning AI setup script (lightning_setup.sh)
- Comprehensive .gitignore
- Requirements files (requirements.txt, requirements-dev.txt)
