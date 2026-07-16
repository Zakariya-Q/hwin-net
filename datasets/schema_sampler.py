"""
HWIN-Net: Schema-Aware Samplers

Mathematical Purpose
--------------------
Implements stratified sampling over schemas (O, a) per Datasets section
of HWIN_Net_Spec.md. Ensures balanced training across:
- Different platforms a in A
- Different observation sets O subset V
- Different |O| (observation cardinalities)

Theory Traceability
-------------------
- Axiom A3 (Non-degenerate Heterogeneity): variable |O|
- Axiom A2 (Schema Action): platform-specific Phi_a
- Definition D1 (Schema): g = (O, a)
- Axiom A4 (Uniform Identifiability): r_0(g) threshold
"""

import torch
from torch.utils.data import Sampler
from typing import List, Dict, Iterator, Optional, Tuple
import numpy as np
from collections import defaultdict
import random


class SchemaSampler(Sampler):
    """
    Base schema-aware sampler.
    
    Groups samples by schema (O, a) and samples proportionally.
    """
    
    def __init__(
        self,
        dataset,
        schema_key_fn: callable,
        batch_size: int,
        drop_last: bool = False,
        shuffle: bool = True,
        seed: int = 42,
    ):
        """
        Args:
            dataset: Dataset with __getitem__ returning M_O and a_idx
            schema_key_fn: Function (idx) -> schema_key string
            batch_size: Batch size
            drop_last: Whether to drop incomplete batches
            shuffle: Whether to shuffle
            seed: Random seed
        """
        self.dataset = dataset
        self.schema_key_fn = schema_key_fn
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        
        # Build schema index
        self.schema_indices = defaultdict(list)
        self._build_schema_index()
        
        self.num_batches = self._compute_num_batches()
    
    def _build_schema_index(self):
        """Group sample indices by schema."""
        for idx in range(len(self.dataset)):
            key = self.schema_key_fn(idx)
            self.schema_indices[key].append(idx)
        
        # Convert to lists
        self.schema_indices = {
            k: list(v) for k, v in self.schema_indices.items()
        }
    
    def _compute_num_batches(self) -> int:
        total = len(self.dataset)
        if self.drop_last:
            return total // self.batch_size
        else:
            return (total + self.batch_size - 1) // self.batch_size
    
    def __iter__(self) -> Iterator[List[int]]:
        if self.shuffle:
            # Shuffle within each schema
            rng = random.Random(self.seed)
            for key in self.schema_indices:
                rng.shuffle(self.schema_indices[key])
        
        # Flatten and batch
        all_indices = []
        for key in sorted(self.schema_indices.keys()):
            all_indices.extend(self.schema_indices[key])
        
        # Handle remainder
        if not self.drop_last and len(all_indices) % self.batch_size != 0:
            pass  # Keep as is
        
        for i in range(0, len(all_indices), self.batch_size):
            batch = all_indices[i:i + self.batch_size]
            if len(batch) == self.batch_size or not self.drop_last:
                yield batch
    
    def __len__(self) -> int:
        return self.num_batches


class StratifiedSchemaSampler(SchemaSampler):
    """
    Stratified sampler ensuring balanced schema representation.
    
    Samples equal number of batches from each schema group,
    or proportional to schema frequency.
    """
    
    def __init__(
        self,
        dataset,
        schema_key_fn: callable,
        batch_size: int,
        strategy: str = "balanced",  # "balanced", "proportional", "min_size"
        min_samples_per_schema: int = 1,
        **kwargs,
    ):
        super().__init__(dataset, schema_key_fn, batch_size, **kwargs)
        self.strategy = strategy
        self.min_samples_per_schema = min_samples_per_schema
    
    def __iter__(self) -> Iterator[List[int]]:
        if self.shuffle:
            rng = random.Random(self.seed)
            for key in self.schema_indices:
                rng.shuffle(self.schema_indices[key])
        
        if self.strategy == "balanced":
            yield from self._balanced_iter()
        elif self.strategy == "proportional":
            yield from self._proportional_iter()
        elif self.strategy == "min_size":
            yield from self._min_size_iter()
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def _balanced_iter(self) -> Iterator[List[int]]:
        """Equal representation from each schema."""
        schema_keys = list(self.schema_indices.keys())
        pointers = {k: 0 for k in schema_keys}
        
        while True:
            batch = []
            # Try to get one from each schema
            for key in schema_keys:
                if len(batch) >= self.batch_size:
                    break
                idx = self.schema_indices[key]
                if pointers[key] < len(idx):
                    batch.append(idx[pointers[key]])
                    pointers[key] += 1
            
            if not batch:
                break
            
            # Fill remainder with available samples
            while len(batch) < self.batch_size:
                # Random schema
                key = random.choice(schema_keys)
                if pointers[key] < len(self.schema_indices[key]):
                    batch.append(self.schema_indices[key][pointers[key]])
                    pointers[key] += 1
                else:
                    # Schema exhausted, try others
                    available = [k for k in schema_keys 
                                if pointers[k] < len(self.schema_indices[k])]
                    if not available:
                        break
                    key = random.choice(available)
                    batch.append(self.schema_indices[key][pointers[key]])
                    pointers[key] += 1
            
            if len(batch) == self.batch_size or not self.drop_last:
                yield batch
    
    def _proportional_iter(self) -> Iterator[List[int]]:
        """Proportional to schema frequency."""
        schema_keys = list(self.schema_indices.keys())
        schema_sizes = {k: len(v) for k, v in self.schema_indices.items()}
        total = sum(schema_sizes.values())
        
        # Target proportions
        target_ratios = {k: v / total for k, v in schema_sizes.items()}
        
        pointers = {k: 0 for k in schema_keys}
        
        while True:
            batch = []
            for key in schema_keys:
                target = int(self.batch_size * target_ratios[key])
                for _ in range(target):
                    if len(batch) >= self.batch_size:
                        break
                    if pointers[key] < len(self.schema_indices[key]):
                        batch.append(self.schema_indices[key][pointers[key]])
                        pointers[key] += 1
            
            # Fill rest
            while len(batch) < self.batch_size:
                key = random.choice(schema_keys)
                if pointers[key] < len(self.schema_indices[key]):
                    batch.append(self.schema_indices[key][pointers[key]])
                    pointers[key] += 1
                else:
                    available = [k for k in schema_keys 
                                if pointers[k] < len(self.schema_indices[k])]
                    if not available:
                        break
                    key = random.choice(available)
                    batch.append(self.schema_indices[key][pointers[key]])
                    pointers[key] += 1
            
            if not batch:
                break
            if len(batch) == self.batch_size or not self.drop_last:
                yield batch
    
    def _min_size_iter(self) -> Iterator[List[int]]:
        """Each schema gets at least min_samples_per_schema."""
        schema_keys = list(self.schema_indices.keys())
        pointers = {k: 0 for k in schema_keys}
        
        while True:
            batch = []
            # Minimum from each schema
            for key in schema_keys:
                for _ in range(self.min_samples_per_schema):
                    if len(batch) >= self.batch_size:
                        break
                    if pointers[key] < len(self.schema_indices[key]):
                        batch.append(self.schema_indices[key][pointers[key]])
                        pointers[key] += 1
            
            # Fill rest randomly
            while len(batch) < self.batch_size:
                key = random.choice(schema_keys)
                if pointers[key] < len(self.schema_indices[key]):
                    batch.append(self.schema_indices[key][pointers[key]])
                    pointers[key] += 1
                else:
                    available = [k for k in schema_keys 
                                if pointers[k] < len(self.schema_indices[k])]
                    if not available:
                        break
                    key = random.choice(available)
                    batch.append(self.schema_indices[key][pointers[key]])
                    pointers[key] += 1
            
            if not batch:
                break
            if len(batch) == self.batch_size or not self.drop_last:
                yield batch


class CardinalitySampler(Sampler):
    """
    Sampler stratified by observation cardinality |O|.
    
    Ensures training sees varied |O| values, important for learning
    r_0(g) thresholds (Lemma 7, Theorem 4).
    """
    
    def __init__(
        self,
        dataset,
        cardinality_fn: callable,
        batch_size: int,
        n_cardinality_bins: int = 10,
        **kwargs,
    ):
        self.dataset = dataset
        self.cardinality_fn = cardinality_fn
        self.batch_size = batch_size
        self.n_cardinality_bins = n_cardinality_bins
        
        self.cardinality_bins = defaultdict(list)
        self._build_cardinality_bins()
    
    def _build_cardinality_bins(self):
        for idx in range(len(self.dataset)):
            card = self.cardinality_fn(idx)
            bin_idx = min(card, self.n_cardinality_bins - 1)
            self.cardinality_bins[bin_idx].append(idx)
    
    def __iter__(self) -> Iterator[List[int]]:
        # Round-robin across cardinality bins
        bins = list(self.cardinality_bins.keys())
        pointers = {b: 0 for b in bins}
        
        while True:
            batch = []
            for b in bins:
                if len(batch) >= self.batch_size:
                    break
                if pointers[b] < len(self.cardinality_bins[b]):
                    batch.append(self.cardinality_bins[b][pointers[b]])
                    pointers[b] += 1
            
            # Fill rest
            while len(batch) < self.batch_size:
                available = [b for b in bins 
                            if pointers[b] < len(self.cardinality_bins[b])]
                if not available:
                    break
                b = random.choice(available)
                batch.append(self.cardinality_bins[b][pointers[b]])
                pointers[b] += 1
            
            if not batch:
                break
            if len(batch) == self.batch_size:
                yield batch
    
    def __len__(self) -> int:
        total = len(self.dataset)
        return total // self.batch_size


def create_cardinality_fn(num_variables: int):
    """Create a cardinality function from dataset."""
    def fn(idx):
        item = dataset[idx]
        return int(item['M_O'].sum().item())
    return fn


class SchemaAwareSampler(Sampler):
    """
    High-level schema-aware sampler that combines schema and cardinality stratification.
    
    This is the main sampler class used by HWIN-Net training for balanced
    sampling across platforms and observation cardinalities.
    """
    
    def __init__(
        self,
        dataset,
        batch_size: int = 32,
        shuffle: bool = True,
        seed: int = 42,
        schema_strategy: str = "balanced",
        use_cardinality: bool = True,
        n_cardinality_bins: int = 10,
        min_samples_per_schema: int = 1,
    ):
        """
        Args:
            dataset: Schema-aware dataset with M_O and a_idx
            batch_size: Batch size
            shuffle: Whether to shuffle
            seed: Random seed
            schema_strategy: "balanced", "proportional", or "min_size"
            use_cardinality: Whether to also stratify by |O|
            n_cardinality_bins: Number of cardinality bins
            min_samples_per_schema: Minimum samples per schema per batch
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        
        # Build schema index: (tuple of observed vars, platform) -> indices
        self.schema_indices = defaultdict(list)
        self.cardinality_indices = defaultdict(list)
        
        # Extract schema and cardinality for each sample
        for idx in range(len(self.dataset)):
            item = self.dataset[idx]
            M_O = item['M_O']
            a_idx = int(item['a_idx'].item())
            observed = tuple(int(i) for i in range(len(M_O)) if M_O[i] > 0)
            card = len(observed)
            schema_key = (observed, a_idx)
            
            self.schema_indices[schema_key].append(idx)
            self.cardinality_indices[card].append(idx)
        
        # Convert to lists
        self.schema_indices = {k: list(v) for k, v in self.schema_indices.items()}
        self.cardinality_indices = {k: list(v) for k, v in self.cardinality_indices.items()}
        
        self.use_cardinality = use_cardinality
        self.schema_strategy = schema_strategy
        self.min_samples_per_schema = min_samples_per_schema
        self.n_cardinality_bins = n_cardinality_bins
        
        # Compute total samples and num_batches
        self.total_samples = len(dataset)
        self.num_batches = self.total_samples // self.batch_size
    
    def __iter__(self):
        if self.shuffle:
            rng = random.Random(self.seed)
            for indices in self.schema_indices.values():
                rng.shuffle(indices)
            for indices in self.cardinality_indices.values():
                rng.shuffle(indices)
        
        schema_keys = list(self.schema_indices.keys())
        cardinality_keys = sorted(self.cardinality_indices.keys())
        
        # Strategy: combine schema and cardinality
        if self.use_cardinality and cardinality_keys:
            yield from self._stratified_cardinality_iter(schema_keys, cardinality_keys)
        else:
            yield from self._schema_only_iter(schema_keys)
    
    def _schema_only_iter(self, schema_keys):
        """Iterate with schema stratification only."""
        if self.schema_strategy == "balanced":
            pointers = {k: 0 for k in schema_keys}
            while True:
                batch = []
                for key in schema_keys:
                    if len(batch) >= self.batch_size:
                        break
                    if pointers[key] < len(self.schema_indices[key]):
                        batch.append(self.schema_indices[key][pointers[key]])
                        pointers[key] += 1
                
                # Fill remainder
                while len(batch) < self.batch_size:
                    available = [k for k in schema_keys 
                                if pointers[k] < len(self.schema_indices[k])]
                    if not available:
                        break
                    key = random.choice(available)
                    batch.append(self.schema_indices[key][pointers[key]])
                    pointers[key] += 1
                
                if not batch:
                    break
                if len(batch) == self.batch_size:
                    yield batch
                    
        elif self.schema_strategy == "min_size":
            pointers = {k: 0 for k in schema_keys}
            while True:
                batch = []
                for key in schema_keys:
                    for _ in range(self.min_samples_per_schema):
                        if len(batch) >= self.batch_size:
                            break
                        if pointers[key] < len(self.schema_indices[key]):
                            batch.append(self.schema_indices[key][pointers[key]])
                            pointers[key] += 1
                
                while len(batch) < self.batch_size:
                    available = [k for k in schema_keys 
                                if pointers[k] < len(self.schema_indices[k])]
                    if not available:
                        break
                    key = random.choice(available)
                    batch.append(self.schema_indices[key][pointers[key]])
                    pointers[key] += 1
                
                if not batch:
                    break
                if len(batch) == self.batch_size:
                    yield batch
        else:
            # Proportional
            schema_sizes = {k: len(v) for k, v in self.schema_indices.items()}
            total = sum(schema_sizes.values())
            target_ratios = {k: v / total for k, v in schema_sizes.items()}
            
            pointers = {k: 0 for k in schema_keys}
            while True:
                batch = []
                for key in schema_keys:
                    target = int(self.batch_size * target_ratios[key])
                    for _ in range(target):
                        if len(batch) >= self.batch_size:
                            break
                        if pointers[key] < len(self.schema_indices[key]):
                            batch.append(self.schema_indices[key][pointers[key]])
                            pointers[key] += 1
                
                while len(batch) < self.batch_size:
                    available = [k for k in schema_keys 
                                if pointers[k] < len(self.schema_indices[k])]
                    if not available:
                        break
                    key = random.choice(available)
                    batch.append(self.schema_indices[key][pointers[key]])
                    pointers[key] += 1
                
                if not batch:
                    break
                if len(batch) == self.batch_size:
                    yield batch
    
    def _stratified_cardinality_iter(self, schema_keys, cardinality_keys):
        """Iterate with both schema and cardinality stratification."""
        # Combine: sample by cardinality first, then by schema within cardinality
        pointers_schema = {k: 0 for k in schema_keys}
        pointers_card = {k: 0 for k in cardinality_keys}
        
        while True:
            batch = []
            # Round-robin across cardinalities
            for card in cardinality_keys:
                if len(batch) >= self.batch_size:
                    break
                # Within cardinality, sample balanced across schemas
                # Find schemas that have this cardinality
                card_schemas = [k for k in schema_keys 
                               if len(k[0]) == card and pointers_schema[k] < len(self.schema_indices[k])]
                if card_schemas:
                    key = random.choice(card_schemas)
                    batch.append(self.schema_indices[key][pointers_schema[key]])
                    pointers_schema[key] += 1
                    pointers_card[card] += 1
            
            # Fill remainder
            while len(batch) < self.batch_size:
                available_cards = [c for c in cardinality_keys 
                                  if pointers_card[c] < len(self.cardinality_indices[c])]
                if not available_cards:
                    break
                card = random.choice(available_cards)
                card_schemas = [k for k in schema_keys 
                               if len(k[0]) == card and pointers_schema[k] < len(self.schema_indices[k])]
                if not card_schemas:
                    # Try any schema
                    available_schemas = [k for k in schema_keys 
                                        if pointers_schema[k] < len(self.schema_indices[k])]
                    if not available_schemas:
                        break
                    key = random.choice(available_schemas)
                else:
                    key = random.choice(card_schemas)
                batch.append(self.schema_indices[key][pointers_schema[key]])
                pointers_schema[key] += 1
            
            if not batch:
                break
            if len(batch) == self.batch_size:
                yield batch
    
    def __len__(self):
        return self.num_batches
