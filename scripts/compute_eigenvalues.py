import os
import time
from pathlib import Path
from typing import List, Optional
import fire
import torch
from torch_geometric.data import Data


def compute_eigenvalues_batched(data_list: List[Data]) -> torch.Tensor:
    """Computes linearized shape and scale conditioning features: [s_1, s_2, s_3, S]."""
    covs = []
    
    for data in data_list:
        pos = data.pos
        # Center positions: X_c = X - mean(X)
        pos_centered = pos - pos.mean(dim=0, keepdim=True)
        # Covariance matrix (unnormalized): X_c^T * X_c
        cov = torch.matmul(pos_centered.T, pos_centered) / pos_centered.shape[0]
        covs.append(cov)
            
    covs = torch.stack(covs)  # Shape: (B, 3, 3)
    
    # 1. Unnormalized eigenvalues (\lambda)
    e_unnorm = torch.linalg.eigh(covs).eigenvalues  # Shape: (B, 3)
    # Sort eigenvalues in descending order
    e_unnorm = torch.sort(e_unnorm, descending=True, dim=-1).values
    
    # Clip negative values to zero (due to numerical precision)
    e_unnorm = torch.clamp(e_unnorm, min=0.0)
    
    # 2. Linearization (E_i = \sqrt{\lambda_i})
    E = torch.sqrt(e_unnorm)
    
    # 3. Global scale scalar (S = \sum E_i)
    S = E.sum(dim=-1, keepdim=True)
    
    # 4. Fractional shape normalization (\hat{s}_i = E_i / S)
    # Avoid division by zero
    S_safe = torch.clamp(S, min=1e-8)
    s_hat = E / S_safe
    
    # 5. Feed-Forward Integration (c = [\hat{s}_1, \hat{s}_2, \hat{s}_3, S])
    c = torch.cat([s_hat, S], dim=-1)  # Shape: (B, 4)
    
    return c


def process_file(file_path: str | Path, overwrite: bool = True):
    file_path = Path(file_path)
    if not file_path.exists():
        print(f"WARNING: File not found: {file_path}")
        return

    print(f"Loading {file_path}...")
    start_time = time.time()
    data_list = torch.load(file_path)
    print(f"Loaded {len(data_list)} molecules in {time.time() - start_time:.2f}s")
    
    print("Computing eigenvalues...")
    start_time = time.time()
    eigenvalues_and_scale = compute_eigenvalues_batched(data_list)
    print(f"Computed eigenvalues in {time.time() - start_time:.2f}s")
    
    print("Appending features to dataset, deleting old ones...")
    for i, data in enumerate(data_list):
        for key in ['eigenvalues', 'eigenvalues_normalized', 'eigenvalues_normalized_no_h']:
            if key in data:
                del data[key]
        data.eigenvalues_and_scale = eigenvalues_and_scale[i]
        
    if overwrite:
        save_path = file_path
    else:
        save_path = file_path.parent / f"{file_path.stem}_with_eig.pt"
        
    print(f"Saving to {save_path}...")
    start_time = time.time()
    
    # Safe save: write to a temporary file first, then rename
    temp_save_path = save_path.with_suffix('.pt.tmp')
    torch.save(data_list, temp_save_path)
    os.replace(temp_save_path, save_path)
    
    print(f"Saved in {time.time() - start_time:.2f}s")
    print(f"Successfully processed {file_path}!\n")


def main(
    data_dir: str = "data",
    file_path: Optional[str] = None,
    overwrite: bool = True,
):
    """Computes eigenvalues for molecules in preprocessed datasets and appends them.

    Args:
        data_dir: Path to the data directory containing qm9/ and geom_drugs/.
                  Defaults to 'data'.
        file_path: Optional path to a specific .pt file to process.
        overwrite: Whether to overwrite the input file with the appended features.
    """
    if file_path:
        process_file(file_path, overwrite=overwrite)
        return

    data_root = Path(data_dir)
    
    # If the user passed a specific dataset dir (e.g. data/qm9) instead of the root data/
    if (data_root / "preprocessed").exists():
        target_dirs = [data_root]
    else:
        target_dirs = [data_root / "qm9", data_root / "geom_drugs"]

    processed_any = False
    for target in target_dirs:
        preprocessed_dir = target / "preprocessed"
        if preprocessed_dir.exists():
            for split in ["train", "val", "test"]:
                split_file = preprocessed_dir / f"{split}.pt"
                if split_file.exists():
                    process_file(split_file, overwrite=overwrite)
                    processed_any = True

    if not processed_any:
        print(f"ERROR: No preprocessed datasets found under {data_root}. Please check dataset paths.")


if __name__ == "__main__":
    fire.Fire(main)
