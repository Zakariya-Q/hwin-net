#!/usr/bin/env python
"""
Compute dataset statistics: feature mean/std and PCA basis for manifold retraction.

Run this ONCE after data preprocessing to generate:
- feature_mean.pt: mean of features (100 vars)
- feature_std.pt: std of features (100 vars)  
- pca_basis.pt: PCA basis for mu(M) manifold (64 x pca_components)
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import sys

# We only need pandas/numpy, no hwin_net imports required for this script


def compute_feature_stats(data_path: str, target_column: str = "y", num_variables: int = 100):
    """Compute mean and std of features from training data."""
    print(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)
    
    # Identify feature columns (exclude target, platform, station_id, etc.)
    exclude_cols = {target_column, "platform", "station_id", "schema_id", "timestamp", "a_idx"}
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    feature_cols = feature_cols[:num_variables]
    
    print(f"Found {len(feature_cols)} feature columns: {feature_cols[:5]}...")
    
    features = df[feature_cols].values.astype(np.float32)
    
    # Handle NaN values
    features = np.nan_to_num(features, nan=0.0)
    
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std = np.where(std == 0, 1.0, std)  # Avoid division by zero
    
    return torch.from_numpy(mean), torch.from_numpy(std), feature_cols


def compute_pca_basis(data_path: str, pca_components: int = 32, latent_dim: int = 64):
    """
    Compute PCA basis from a subset of data.
    This is used as initial basis for Manifold Retraction (M3).
    """
    print(f"Computing PCA basis from {data_path}...")
    
    # Load a subset of data for PCA (full 2.8M samples would be too much)
    df = pd.read_parquet(data_path)
    
    exclude_cols = {"y", "platform", "station_id", "schema_id", "timestamp", "a_idx"}
    feature_cols = [c for c in df.columns if c not in exclude_cols][:100]
    
    features = df[feature_cols].values.astype(np.float32)
    features = np.nan_to_num(features, nan=0.0)
    
    N, d = features.shape
    print(f"Data shape: {N} samples x {d} features")
    
    # Sample subset for efficiency
    sample_size = min(50000, N)
    indices = np.random.choice(N, sample_size, replace=False)
    sample_features = features[indices]
    
    # First project to latent_dim
    # Using a random projection matrix
    np.random.seed(42)
    proj = np.random.randn(d, latent_dim) * 0.01
    projected = sample_features @ proj  # (sample_size, latent_dim)
    
    # Then PCA on projected space
    from sklearn.decomposition import PCA
    import warnings
    warnings.filterwarnings("ignore")
    
    pca = PCA(n_components=pca_components)
    pca.fit(projected)
    
    # Basis is principal components (latent_dim, pca_components)
    basis = torch.from_numpy(pca.components_.T).float()  # (latent_dim, pca_components)
    
    print(f"PCA basis shape: {basis.shape}")
    print(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")
    
    return basis


def main():
    parser = argparse.ArgumentParser(description="Compute dataset statistics for HWIN-Net")
    parser.add_argument("--train_data", type=str, default="./data/train.parquet", help="Path to training data")
    parser.add_argument("--output_dir", type=str, default="./data/stats", help="Output directory for stats")
    parser.add_argument("--num_variables", type=int, default=100, help="Number of feature variables")
    parser.add_argument("--pca_components", type=int, default=32, help="Number of PCA components")
    parser.add_argument("--latent_dim", type=int, default=64, help="Latent dimension")
    parser.add_argument("--target_column", type=str, default="y", help="Target column name")
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Compute feature mean/std
    print("=" * 60)
    print("COMPUTING FEATURE STATISTICS")
    print("=" * 60)
    
    mean, std, feature_cols = compute_feature_stats(
        args.train_data, 
        args.target_column, 
        args.num_variables
    )
    
    torch.save(mean, output_dir / "feature_mean.pt")
    torch.save(std, output_dir / "feature_std.pt")
    
    print(f"Saved feature_mean.pt: {mean.shape}")
    print(f"Saved feature_std.pt: {std.shape}")
    print(f"Mean range: [{mean.min():.4f}, {mean.max():.4f}]")
    print(f"Std range: [{std.min():.4f}, {std.max():.4f}]")
    
    # Compute PCA basis (initial - will be updated during training)
    print("\n" + "=" * 60)
    print("COMPUTING INITIAL PCA BASIS")
    print("=" * 60)
    
    basis = compute_pca_basis(
        args.train_data,
        args.pca_components,
        args.latent_dim
    )
    
    torch.save(basis, output_dir / "pca_basis.pt")
    print(f"Saved pca_basis.pt: {basis.shape}")
    
    # Save feature column names for reference
    with open(output_dir / "feature_columns.txt", "w") as f:
        for col in feature_cols:
            f.write(f"{col}\n")
    
    print("\n" + "=" * 60)
    print("STATISTICS COMPUTATION COMPLETE")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"  - feature_mean.pt")
    print(f"  - feature_std.pt")
    print(f"  - pca_basis.pt")
    print(f"  - feature_columns.txt")
    print("\nUpdate config.yaml with:")
    print(f"  data.feature_mean_path: ./data/stats/feature_mean.pt")
    print(f"  data.feature_std_path: ./data/stats/feature_std.pt")
    print(f"  retraction.manifold_basis_path: ./data/stats/pca_basis.pt")


if __name__ == "__main__":
    main()
