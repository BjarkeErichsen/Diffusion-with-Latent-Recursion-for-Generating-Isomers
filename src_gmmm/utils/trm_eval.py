import os
import uuid
import torch
import shutil
from typing import Dict, Any

import sys

def save_trm_eval_data(data: Dict[str, Any], epoch: int = -1):
    """Dumps accumulated diffusion tracking data to disk, clearing previous trajectories to save space."""
    if not data["X_t"]["pos"] and not data["X_t"]["h"]:
        return

    try:
        from hydra.core.hydra_config import HydraConfig
        run_dir = HydraConfig.get().runtime.output_dir
    except Exception:
        # Fallback to current working directory if Hydra is not initialized
        run_dir = os.getcwd()
        
    base_dir = os.path.join(run_dir, "val_trm_trajectories")
    
    # Clear previous results to ensure we only keep the latest batch and save disk space
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    os.makedirs(base_dir, exist_ok=True)
    
    batch_id = str(uuid.uuid4())[:8]
    epoch_str = f"epoch_{epoch}_" if epoch != -1 else ""
    
    # Save with epoch and batch identifiers
    torch.save(data["X_t"], os.path.join(base_dir, f"X_t_{epoch_str}batch_{batch_id}.pt"))
    
    if "X_1" in data:
        torch.save(data["X_1"], os.path.join(base_dir, f"X_1_{epoch_str}batch_{batch_id}.pt"))
    elif "X_0_pred" in data: # Backward compatibility
        torch.save(data["X_0_pred"], os.path.join(base_dir, f"X_1_{epoch_str}batch_{batch_id}.pt"))

    if "val_datapoint" in data and data["val_datapoint"] is not None:
        torch.save(data["val_datapoint"], os.path.join(base_dir, f"val_datapoint_{epoch_str}batch_{batch_id}.pt"))
    
    # Transpose the Z dimensional array: Extract a single list containing all timestep states for each inner iteration
    num_iters = len(data["Z"][0]) if data["Z"] and data["Z"][0] else 0
    for i in range(num_iters):
        z_i_list = []
        for t_idx in range(len(data["Z"])):
            if i < len(data["Z"][t_idx]):
                z_i_list.append(data["Z"][t_idx][i])
        torch.save(z_i_list, os.path.join(base_dir, f"Z_{i}_{epoch_str}batch_{batch_id}.pt"))