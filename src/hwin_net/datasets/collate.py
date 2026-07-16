"""
HWIN-Net: Collate Functions

Mathematical Purpose
--------------------
Implements proper batching for schema-aware data:
- Pad observation masks
- Stack platform indices
- Handle variable |O| per sample
- Optionally include schema metadata

Theory Traceability
-------------------
- Definition D2 (Observation): z_g = pi_O o Phi_a(s)
- Axiom A3 (Non-degenerate Heterogeneity): variable |O| per sample
"""

import torch
from typing import Dict, List, Tuple, Optional, Any
from torch.utils.data import default_collate


def hwin_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for HWIN-Net batches.
    
    Args:
        batch: List of dicts with keys x, M_O, a_idx, (y)
    
    Returns:
        Dict with batched tensors:
        - x: [B, n]
        - M_O: [B, n]
        - a_idx: [B]
        - y: [B, ...] if present
    """
    if not batch:
        raise ValueError("Empty batch")
    
    # Check format
    first = batch[0]
    if isinstance(first, dict):
        return _collate_dict_batch(batch)
    elif isinstance(first, (list, tuple)):
        return _collate_tuple_batch(batch)
    else:
        raise TypeError(f"Unexpected batch element type: {type(first)}")


def _collate_dict_batch(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate list of dicts."""
    # Get all keys
    keys = batch[0].keys()
    
    # Stack tensors for each key
    out = {}
    for key in keys:
        values = [item[key] for item in batch]
        
        if isinstance(values[0], torch.Tensor):
            out[key] = torch.stack(values, dim=0)
        elif isinstance(values[0], (int, float)):
            out[key] = torch.tensor(values)
        else:
            # Try default collate
            out[key] = default_collate(values)
    
    return out


def _collate_tuple_batch(batch: List[Tuple]) -> Dict[str, torch.Tensor]:
    """Collate list of tuples (x, M_O, a_idx, y?)."""
    # Transpose
    transposed = list(zip(*batch))
    
    # Stack
    x = torch.stack(transposed[0], dim=0)
    M_O = torch.stack(transposed[1], dim=0)
    a_idx = torch.stack(transposed[2], dim=0) if len(transposed) > 2 else None
    y = torch.stack(transposed[3], dim=0) if len(transposed) > 3 else None
    
    out = {
        'x': x,
        'M_O': M_O,
        'a_idx': a_idx,
    }
    if y is not None:
        out['y'] = y
    
    return out


def hwin_collate_with_schema(
    batch: List[Dict],
    include_schema: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Collate with optional schema information.
    
    If samples have 'schema_id' or 'schema_O', includes them in output.
    """
    out = hwin_collate_fn(batch)
    
    if include_schema and 'schema_id' in batch[0]:
        out['schema_id'] = [item['schema_id'] for item in batch]
    
    if include_schema and 'schema_O' in batch[0]:
        # schema_O is list of observed variable indices
        schema_O = [item['schema_O'] for item in batch]
        # Pad to max length
        max_len = max(len(s) for s in schema_O)
        schema_O_padded = torch.full(
            (len(batch), max_len), -1, dtype=torch.long
        )
        for i, s in enumerate(schema_O):
            schema_O_padded[i, :len(s)] = torch.tensor(s)
        out['schema_O'] = schema_O_padded
    
    return out


def hwin_collate_mixed_precision(
    batch: List[Dict],
    dtype: torch.dtype = torch.float32,
) -> Dict[str, torch.Tensor]:
    """Collate with specific dtype for mixed precision."""
    out = hwin_collate_fn(batch)
    
    # Convert float tensors
    for key, val in out.items():
        if isinstance(val, torch.Tensor) and val.is_floating_point():
            out[key] = val.to(dtype)
    
    return out
