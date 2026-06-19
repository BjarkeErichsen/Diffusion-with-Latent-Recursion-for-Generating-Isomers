import os
import sys
import random
import json
import torch
from pathlib import Path
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src_gmmm.data.dataset import Dataset
from src_gmmm.metrics.geom_drugs import GeomDrugsMetrics
from src_gmmm.data.utils import atoms_from_tensors

def main(remove_h: bool = True):
    base_path = Path("data/geom_drugs/preprocessed")
    
    print("Loading GEOM datasets...")
    train_data = Dataset(base_path / "train.pt")
    val_data = Dataset(base_path / "val.pt")
    test_data = Dataset(base_path / "test.pt")
    
    len_train = len(train_data)
    len_val = len(val_data)
    len_test = len(test_data)
    total_len = len_train + len_val + len_test
    
    subset_size = max(1, total_len // 20)
    print(f"Total size: {total_len}, Subset size: {subset_size} (1/20th)")
    
    random.seed(42)
    indices = random.sample(range(total_len), subset_size)
    indices.sort()
    
    subset_data = []
    print("Sampling 1/20th of the dataset...")
    for idx in tqdm(indices, desc="Fetching data"):
        if idx < len_train:
            subset_data.append(train_data[idx])
        elif idx < len_train + len_val:
            subset_data.append(val_data[idx - len_train])
        else:
            subset_data.append(test_data[idx - len_train - len_val])
            
    # Initialize the metric precisely as used during training.
    # We set remove_h based on the explicit argument:
    metrics = GeomDrugsMetrics(
        remove_h=remove_h,
        json_path=base_path / "train_infos.json"
    )
    
    decoder = ["H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Br", "I", "Bi"]
    
    print("Computing metrics...")
    for data in tqdm(subset_data, desc="Processing molecules"):
        # We process one graph at a time
        h_atomic_numbers = data.h
        pos = data.pos
        
        # 1. Remove hydrogens explicitly if the setting is True
        if remove_h:
            mask = h_atomic_numbers != 1
            h_atomic_numbers = h_atomic_numbers[mask]
            pos = pos[mask]
            
            # If the molecule is empty after removing H, skip
            if h_atomic_numbers.shape[0] == 0:
                continue
            
        # 2. Map atomic numbers to decoder indices
        atomic_numbers = [1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53, 83]
        mapping = {v: i for i, v in enumerate(atomic_numbers)}
        h = torch.tensor([mapping[hi.item()] for hi in h_atomic_numbers])
        
        # Create a single pointer for a single graph
        ptr = torch.tensor([0, h.shape[0]])
        
        # Convert to ase.Atoms using the same logic as the training loop
        atoms_list = atoms_from_tensors(h, pos, ptr, decoder)
        
        # The update function internally ignores 'H', constructs a heavy-atom RWMol,
        # then re-adds hydrogens using Chem.AddHs().
        metrics.update(atoms_list)
        
    summary = metrics.summarize()
    
    # Log to tmp directory
    os.makedirs("tmp", exist_ok=True)
    out_path = "tmp/geom_metrics_test_stats.json"
    with open(out_path, "w") as f:
        # Convert numpy types to native python types for JSON serialization if necessary
        def convert_np(obj):
            if hasattr(obj, "item"): return obj.item()
            elif hasattr(obj, "tolist"): return obj.tolist()
            return obj
            
        json.dump({k: convert_np(v) for k,v in summary.items()}, f, indent=4)
        
    print(f"Stats saved to {out_path}")
    for k, v in summary.items():
        if not isinstance(v, list) and not isinstance(v, dict):
            print(f"{k}: {v}")

if __name__ == "__main__":
    # You can toggle this to True to explicitly remove hydrogens before computing metrics
    main(remove_h=False)
