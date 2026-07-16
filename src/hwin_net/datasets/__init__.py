"""
HWIN-Net: Dataset Package
"""
from .dataset import HWINSchemaDataset, create_hwin_dataset
from .collate import hwin_collate_fn, hwin_collate_with_schema, hwin_collate_mixed_precision
from .schema_sampler import SchemaAwareSampler

import torch
from torch.utils.data import DataLoader
from typing import Optional


def create_dataloaders(config, split: str = "all"):
    """
    Create train/val/test dataloaders from config.
    
    Args:
        config: Config object
        split: Which split to return (train, val, test, or all)
    
    Returns:
        If split='all': (train_loader, val_loader, test_loader)
        Else: single dataloader
    """
    # Import distributed if available
    try:
        import torch.distributed as dist
        is_distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if is_distributed else 1
        rank = dist.get_rank() if is_distributed else 0
    except:
        is_distributed = False
        world_size = 1
        rank = 0
    
    # Create datasets
    train_dataset = create_hwin_dataset(config, "train")
    val_dataset = create_hwin_dataset(config, "val")
    test_dataset = create_hwin_dataset(config, "test")
    
    # Create samplers
    if is_distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, num_replicas=world_size, rank=rank, shuffle=False
        )
        test_sampler = torch.utils.data.distributed.DistributedSampler(
            test_dataset, num_replicas=world_size, rank=rank, shuffle=False
        )
        shuffle_train = False
        shuffle_val = False
    else:
        train_sampler = None
        val_sampler = None
        test_sampler = None
        shuffle_train = True
        shuffle_val = False
    
    # Optionally use schema-aware sampler
    if config.data.schema_sampler_type == "stratified":
        from .schema_sampler import SchemaAwareSampler
        train_sampler = SchemaAwareSampler(
            train_dataset,
            batch_size=config.training.batch_size,
            shuffle=shuffle_train,
        )
        shuffle_train = False
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        sampler=train_sampler,
        shuffle=shuffle_train and train_sampler is None,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
        collate_fn=hwin_collate_fn,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.val_batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
        collate_fn=hwin_collate_fn,
        drop_last=False,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.val_batch_size,
        sampler=test_sampler,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=config.training.pin_memory,
        collate_fn=hwin_collate_fn,
        drop_last=False,
    )
    
    if split == "train":
        return train_loader
    elif split == "val":
        return val_loader
    elif split == "test":
        return test_loader
    else:
        return train_loader, val_loader, test_loader


__all__ = [
    "HWINSchemaDataset",
    "create_hwin_dataset",
    "hwin_collate_fn",
    "hwin_collate_with_schema",
    "hwin_collate_mixed_precision",
    "SchemaAwareSampler",
    "create_dataloaders",
]
