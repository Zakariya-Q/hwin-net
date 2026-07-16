# HWIN-Net: Hierarchical Water Intelligence Network

**Research implementation.** A theory-derived neural architecture for spatiotemporal water quality prediction across heterogeneous monitoring platforms.

## Overview

HWIN-Net implements a mathematically principled architecture derived from the Schema Identification System (SIS) framework. The model decomposes water quality prediction into six composable modules:

- **M1 Schema Encoder**: Platform-specific observation encoding with masking
- **M2 Recovery Module**: Equivariant recovery maps with weight tying  
- **M3 Manifold Retraction**: Projection onto physically admissible manifold
- **M4 Query Head**: Prediction with aleatoric/epistemic uncertainty
- **M5 Identifiability Gate**: Hard refusal for unidentifiable schemas
- **M6 No-Leakage Regularizer**: Adversarial platform invariance

## Installation

`ash
# From source (editable)
git clone https://github.com/Zakariya-Q/hwin-net.git
cd hwin-net
pip install -e .

# With dev dependencies
pip install -e .[dev]

# With plotting dependencies
pip install -e .[plotting]
`

## Quick Start

`python
import torch
from hwin_net import Config, load_config
from hwin_net.models import HWINNet
from hwin_net.datasets import create_dataloaders
from hwin_net.training import Train_HWIN

# Load config
config = load_config("configs/config.yaml")

# Create data loaders
train_loader, val_loader, test_loader = create_dataloaders(config)

# Create model
model = HWINNet(config)

# Train
trainer = Train_HWIN(model, config)
trainer.fit(train_loader, val_loader)
`

Or use the training script:

`ash
python scripts/train.py --config configs/config.yaml --seed 42 --output_dir ./outputs
`

## Repository Layout

`
hwin-net/
├── README.md
├── LICENSE
├── CITATION.cff
├── CHANGELOG.md
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── SECURITY.md
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
├── .gitignore
├── configs/              # YAML configurations (Hydra-compatible)
├── scripts/              # Executable scripts (train.py, compute_stats.py, etc.)
├── examples/             # Jupyter notebooks and examples
├── tests/                # Pytest test suite
├── docs/                 # Documentation
├── tools/                # Internal tools
├── .github/
│   └── workflows/        # GitHub Actions CI
├── hwin_net/             # Python package (src layout)
│   ├── __init__.py
│   ├── models/           # M1-M6 + HWINNet
│   ├── datasets/         # Data loading & schema-aware sampling
│   ├── losses/           # L_pred, L_rec, L_noleak, L_equiv, L_complex
│   ├── training/         # TTUR optimizer, schedulers, trainer
│   ├── inference/        # Inference engine, uncertainty quantification
│   └── utils/            # Config, seeding, logging, metrics
└── lightning_setup.sh    # Lightning AI environment setup
`

## Architecture

### Modules

| Module | Class | Theory Reference |
|--------|-------|------------------|
| M1 Schema Encoder | SchemaEncoder | Axiom A2, A3, D2 |
| M2 Recovery | RecoveryModule | Axiom A4, Lemma 2, Lemma 5, CC3 |
| M3 Manifold Retraction | ManifoldRetraction | Axiom A1, Theorem 1, D12, CC4 |
| M4 Query Head | QueryHead | Axiom A5, Theorem 3, Lemma 8 |
| M5 Identifiability Gate | IdentifiabilityGate | Axiom A4, Theorem 4, Lemma 7, D11, CC2 |
| M6 No-Leakage | NoLeakageRegularizer | Axiom A5, CC5, Conjecture C1 |

### Tensor Flow

`
x [B, n] + M_O [B, n] + a_idx [B]
    │
    ▼
M1: SchemaEncoder ──▶ z_g [B, k] (masked), h_a [B, k] (full)
    │
    ▼
M2: RecoveryModule ──▶ μ̂ [B, d] (equivariant recovery)
    │
    ▼
M3: ManifoldRetraction ──▶ μ_final [B, d] (on μ(M))
    │
    ▼
M4: QueryHead ──▶ q̂ [B, |Y|], σ²_aleat [B, |Y|]
    │
    ▼
M5: IdentifiabilityGate ──▶ routed [B], r₀ [B]
    │
    ▼
Output: prediction + uncertainty + routing decision
`

## Configuration

All configuration via YAML (Hydra-compatible):

`yaml
# configs/config.yaml
experiment_name: "hwin_net"
log_level: "info"

encoder:
  enabled: true
  encoder_type: "mlp"
  output_dim: 128
  num_platforms: 4
  # ...

recovery:
  enabled: true
  recovery_type: "equivariant_mlp"
  latent_dim: 64
  # ...

# Full config at configs/config.yaml
`

Override from CLI:
`ash
python scripts/train.py --config configs/config.yaml --override "encoder.output_dim=256" "training.max_epochs=200"
`

## Training

### TTUR Optimizer

Uses Two Timescale Update Rule (TTUR) with separate learning rates:
- Encoder/Recovery: encoder_lr
- Gate/Retraction/Query: gate_lr (typically 10x smaller)

### Loss Components

| Loss | Weight | Config |
|------|--------|--------|
| L_pred | λ_pred | loss.lambda_pred |
| L_rec | λ_rec | loss.lambda_rec |
| L_noleak | λ_noleak | loss.lambda_noleak |
| L_equiv | λ_equiv | loss.lambda_equiv |
| L_complex | λ_complex | loss.lambda_complex |

### Checkpointing

Automatic checkpointing saves:
- est.ckpt - Best validation metric
- latest.ckpt - Latest epoch
- epoch_{N}.ckpt - Periodic saves

Resuming:
`ash
python scripts/train.py --config configs/config.yaml --resume outputs/checkpoints/latest.ckpt
`

## Evaluation

`python
from hwin_net.inference import create_inference
from hwin_net.utils import set_seed

set_seed(42)
model.load_state_dict(torch.load("outputs/checkpoints/best.ckpt"))
inference = create_inference(model)

# Single batch
output = inference(x, M_O, a_idx)
print(output.prediction)
print(output.sigma2_total)
print(output.routed)  # 1=routed, 0=refused
`

### Metrics

- **Regression**: NMSE, NLL
- **Calibration**: ECE, MCE, TACE, Sharpness
- **Gate**: RoutingAccuracy, ThresholdAccuracy
- **Recovery**: RecoveryError
- **Equivariance**: EquivarianceError
- **Manifold**: Idempotence, Fixing
- **Uncertainty**: Decomposition (aleatoric/epistemic/non-identifiable)

## Lightning AI

The repository includes a complete Lightning AI setup:

`ash
# On Lightning AI Studio
git clone https://github.com/Zakariya-Q/hwin-net.git
cd hwin-net
bash lightning_setup.sh

# Verifies:
# - Git clone
# - pip install -e .
# - GPU/CUDA availability
# - PyTorch CUDA
# - Dataset availability
# - Ready-to-train check
`

## Reproducibility

All experiments use:
- Fixed seeds: [42, 123, 456, 789, 999]
- Official GroupKFold by station_id
- Official HWIN-Bench preprocessing
- Frozen architecture, loss, hyperparameters

See REPRODUCIBILITY_MANIFEST.md and configs/config.yaml.

## Citation

`ibtex
@software{hwin_net_2025,
  author = {HWIN Team},
  title = {HWIN-Net: Hierarchical Water Intelligence Network},
  version = {0.1.0},
  year = {2025},
  url = {https://github.com/Zakariya-Q/hwin-net}
}
`

See CITATION.cff for full citation metadata.

## Known Limitations

1. **Small-data regime**: Performance on very few observations per station is limited
2. **Extrapolation**: No guarantees for variables/platforms outside training distribution
3. **Computational cost**: Full model ~60K parameters, moderate GPU memory
4. **Platform imbalance**: Assumes balanced platform representation in training data

## Roadmap

- [ ] v0.2.0: Multi-target prediction (vector Y)
- [ ] v0.2.0: Temporal convolutions for sequence modeling
- [ ] v0.3.0: Foundation model pretraining
- [ ] v0.3.0: Zero-shot transfer to new platforms
- [ ] v1.0.0: Publication-ready with full benchmark suite

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Code of Conduct

See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## Security

See [SECURITY.md](SECURITY.md) for vulnerability reporting.
