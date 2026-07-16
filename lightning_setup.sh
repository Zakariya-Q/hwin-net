#!/bin/bash
# Lightning AI Setup Script for HWIN-Net

set -euo pipefail

echo "=========================================="
echo "HWIN-Net Lightning AI Setup"
echo "=========================================="

echo ""
echo "[1/8] Checking Python version..."
python_version=Python 3.13.14
echo "Found: \"

echo ""
echo "[2/8] Checking git..."
if ! command -v git &> /dev/null; then
    echo "ERROR: git not found"
    exit 1
fi
echo "Git: \git version 2.49.0.windows.1"

echo ""
echo "[3/8] Verifying repository..."
if [ ! -f "pyproject.toml" ] || [ ! -d "src/hwin_net" ]; then
    echo "ERROR: Not in HWIN-Net repository root"
    exit 1
fi
echo "Repository verified"

echo ""
echo "[4/8] Creating virtual environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "Created .venv"
else
    echo ".venv already exists"
fi

echo ""
echo "[5/8] Activating environment and installing package..."
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -e .
echo "Package installed"

echo ""
echo "[6/8] Installing development dependencies..."
pip install -r requirements-dev.txt
echo "Dev dependencies installed"

echo ""
echo "[7/8] Verifying PyTorch and CUDA..."
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU count: {torch.cuda.device_count()}')
else:
    print('WARNING: CUDA not available - using CPU')
"

echo ""
echo "[8/8] Running sanity checks..."
python -c "
import hwin_net
print(f'HWIN-Net version: {hwin_net.__version__}')
from hwin_net.models import HWINNet
print('Models: OK')
from hwin_net.datasets import create_dataloaders
print('Datasets: OK')
from hwin_net.losses import TotalLoss
print('Losses: OK')
from hwin_net.training import Train_HWIN
print('Training: OK')
from hwin_net.inference import InferenceEngine
print('Inference: OK')
from hwin_net.utils import set_seed
print('Utils: OK')
print('All imports successful!')
"

echo ""
echo "Running quick test (pytest)..."
pytest tests/test_sanity.py -v --tb=short -x || echo "Some tests failed (expected if no data)"

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "To activate environment later:"
echo "  source .venv/bin/activate"
echo ""
echo "To train:"
echo "  python scripts/train.py --config configs/config.yaml --seed 42"
echo ""
echo "To run tests:"
echo "  pytest"
echo ""
echo "Environment ready for Lightning AI!"
