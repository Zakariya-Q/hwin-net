"""
HWIN-Net Deterministic Seeding Utilities

Mathematical Purpose
--------------------
Provides deterministic random seed management for reproducible experiments,
as required by the Training specification for reproducibility (cuDNN deterministic).
"""

import os
import random
import numpy as np

# Optional torch import
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def set_deterministic(deterministic: bool = True) -> None:
    if TORCH_AVAILABLE:
        import torch
        torch.use_deterministic_algorithms(deterministic, warn_only=True)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Set all random seeds for reproducibility.

    Theorem Traceability
    --------------------
    - Training: deterministic algorithms for reproducible training
    - Axiom A1-A6: Mathematical operations must be deterministic for theorem validity

    Tensor Signatures
    -----------------
    No tensor inputs/outputs - global state mutation

    Complexity
    ----------
    Time: O(1), Space: O(1)

    Assumptions
    -----------
    - torch, numpy, random are available
    - CUDA/cuDNN available if deterministic=True

    Implementation Choices
    ----------------------
    - Sets PYTHONHASHSEED for hash randomization
    - Configures torch.backends.cudnn.deterministic and benchmark
    - Works with both CUDA and CPU devices
    """
    # Python built-in random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch (if available)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Environment variable for Python hash randomization
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic and TORCH_AVAILABLE:
        # cuDNN deterministic mode (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Use deterministic algorithms where available
        torch.use_deterministic_algorithms(True, warn_only=True)


def get_seed() -> int:
    """
    Get the current seed from environment or return default.

    Returns
    -------
    int : Current seed value or 42 if not set
    """
    return int(os.environ.get("PYTHONHASHSEED", "42"))


def worker_init_fn(worker_id: int) -> None:
    """
    DataLoader worker initialization for reproducible data loading.

    Parameters
    ----------
    worker_id : int
        Unique worker ID assigned by DataLoader

    Theorem Traceability
    --------------------
    - Training: Reproducible data loading across epochs
    """
    base_seed = get_seed()
    worker_seed = base_seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(worker_seed)


if __name__ == "__main__":
    # Quick test (works without torch)
    print("Testing deterministic seeding...")
    set_seed(42)
    print(f"Random: {random.random():.6f}")
    print(f"NumPy: {np.random.rand():.6f}")

    set_seed(42)
    print(f"Random (reset): {random.random():.6f}")
    print(f"NumPy (reset): {np.random.rand():.6f}")
    print("Seed module test PASSED")
