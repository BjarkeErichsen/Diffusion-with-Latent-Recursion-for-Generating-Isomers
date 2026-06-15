import os
import time
from pathlib import Path
from typing import List, Optional
import fire
import torch
from torch_geometric.data import Data


def compute_eigenvalues_batched(data_list: List[Data]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Computes unnormalized eigenvalues, normalized eigenvalues, and normalized eigenvalues without hydrogens
    for a list of PyG Data objects.
    """
    covs = []
    counts = []
    
    covs_no_h = []
    counts_no_h = []
    
    for data in data_list:
        pos = data.pos
        # Center positions: X_c = X - mean(X)
        pos_centered = pos - pos.mean(dim=0, keepdim=True)
        # Covariance matrix (unnormalized): X_c^T * X_c
        cov = torch.matmul(pos_centered.T, pos_centered)
        covs.append(cov)
        counts.append(pos.size(0))
        
        # Without hydrogens: H has atomic number 1
        mask_no_h = data.h != 1
        pos_no_h = pos[mask_no_h]
        if pos_no_h.size(0) > 0:
            pos_no_h_centered = pos_no_h - pos_no_h.mean(dim=0, keepdim=True)
            cov_no_h = torch.matmul(pos_no_h_centered.T, pos_no_h_centered)
            covs_no_h.append(cov_no_h)
            counts_no_h.append(pos_no_h.size(0))
        else:
            # Fallback if no heavy atoms exist
            covs_no_h.append(torch.zeros((3, 3), device=pos.device))
            counts_no_h.append(1)
            
    covs = torch.stack(covs)  # Shape: (B, 3, 3)
    counts = torch.tensor(counts, dtype=torch.float32)  # Shape: (B,)
    
    covs_no_h = torch.stack(covs_no_h)  # Shape: (B, 3, 3)
    counts_no_h = torch.tensor(counts_no_h, dtype=torch.float32)  # Shape: (B,)
    
    # 1. Unnormalized eigenvalues
    e_unnorm = torch.linalg.eigh(covs).eigenvalues  # Shape: (B, 3)
    # Sort eigenvalues in descending order
    e_unnorm = torch.sort(e_unnorm, descending=True, dim=-1).values
    
    # 2. Normalized eigenvalues (divided by N, the number of atoms)
    covs_norm = covs / counts.view(-1, 1, 1)
    e_norm = torch.linalg.eigh(covs_norm).eigenvalues  # Shape: (B, 3)
    e_norm = torch.sort(e_norm, descending=True, dim=-1).values
    
    # 3. Normalized eigenvalues without hydrogens (divided by N_heavy)
    covs_norm_no_h = covs_no_h / counts_no_h.view(-1, 1, 1)
    e_norm_no_h = torch.linalg.eigh(covs_norm_no_h).eigenvalues  # Shape: (B, 3)
    e_norm_no_h = torch.sort(e_norm_no_h, descending=True, dim=-1).values
    
    # Clip negative values to zero (due to numerical precision)
    e_unnorm = torch.clamp(e_unnorm, min=0.0)
    e_norm = torch.clamp(e_norm, min=0.0)
    e_norm_no_h = torch.clamp(e_norm_no_h, min=0.0)
    
    return e_unnorm, e_norm, e_norm_no_h


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
    e_unnorm, e_norm, e_norm_no_h = compute_eigenvalues_batched(data_list)
    print(f"Computed eigenvalues in {time.time() - start_time:.2f}s")
    
    print("Appending features to dataset...")
    for i, data in enumerate(data_list):
        data.eigenvalues = e_unnorm[i]
        data.eigenvalues_normalized = e_norm[i]
        data.eigenvalues_normalized_no_h = e_norm_no_h[i]
        
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
