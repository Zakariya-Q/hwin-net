"""
Training entry point for HWIN-Net.
Called via: python -m training.train --config configs/config.yaml --seed 42 --output_dir ./outputs
"""
import os
import sys
import argparse
import yaml
import torch
import numpy as np
import random
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hwin_net.models.hwin_net import HWINNet
from hwin_net.utils.config import Config, load_config
from hwin_net.utils.seed import set_seed
from hwin_net.utils.structured_logging import setup_logging, get_logger
from hwin_net.datasets import create_dataloaders
from hwin_net.training.trainer import Train_HWIN


def parse_args():
    parser = argparse.ArgumentParser(description="Train HWIN-Net")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config YAML")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--override", type=str, action="append", default=[], help="Config overrides (key=value)")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    return parser.parse_args()


def apply_overrides(config, overrides):
    """Apply config overrides from command line."""
    for override in overrides:
        if "=" not in override:
            continue
        key, value = override.split("=", 1)
        # Parse value
        try:
            value = yaml.safe_load(value)
        except:
            pass
        # Set nested config
        keys = key.split(".")
        obj = config
        for k in keys[:-1]:
            obj = getattr(obj, k)
        setattr(obj, keys[-1], value)


def main():
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Apply overrides
    apply_overrides(config, args.override)
    
    # Override seed if provided
    config.training.seed = args.seed
    config.training.deterministic = True
    
    # Set seed
    set_seed(config.training.seed)
    
    # Setup logging
    log_dir = Path(args.output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(log_level=config.log_level, log_file=str(log_dir / "train.log"))
    logger = get_logger(__name__)

    logger.info(f"Starting training with seed {args.seed}")
    logger.info(f"Output dir: {args.output_dir}")
    logger.info(f"Config: {args.config}")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_dataloaders(config)
    
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Create model
    model = HWINNet(config)
    
    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Run training
    results = Train_HWIN(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        logger=logger,
    )
    
    # Save results
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / "metrics.json", "w") as f:
        import json
        json.dump(results, f, indent=2, default=str)
    
    logger.info("Training completed successfully")
    print("Training completed!")
    print(f"Results saved to {output_path / 'metrics.json'}")


if __name__ == "__main__":
    main()
