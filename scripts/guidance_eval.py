import os
import sys
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from omegaconf import DictConfig
import hydra
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
import fire

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src_gmmm.data.dataset import Dataset
from src_gmmm.lit.module import LitModule
from src_gmmm.metrics.geom_drugs import GeomDrugsMetrics
from src_gmmm.data.utils import atoms_from_tensors
from src_gmmm.utils import utils
from scripts.compute_eigenvalues import compute_eigenvalues_batched

TARGET_SHAPES = {
    "Planar": torch.tensor([0.5, 0.5, 0.0]),
    "Linear": torch.tensor([1.0, 0.0, 0.0]),
    "Round": torch.tensor([1/3, 1/3, 1/3]),
}

METHODS = ["CFG", "CG Std", "CG Self"]


def evaluate_shape_condition(lit_module, dataloader, target_name, target_eig, scale, method, device, json_infos_path, predict_scale, n_integration_steps, plot_dir, lambda_scale_coef, thinking_steps):
    all_mses_stable_3d = []
    all_mses_valid_3d = []
    all_mses_all_3d = []
    all_stabilities = []
    all_validities = []
    
    # We will initialize metrics explicitly 
    metrics = GeomDrugsMetrics(remove_h=False, json_path=json_infos_path)
    
    images_saved = False
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Eval {method} | Shape {target_name} | scale {scale}", leave=False)):
        batch = batch.to(device)
        num_graphs = batch.num_graphs
        
        # 2. Construct conditioning vector
        # Target eig is shape (3,) -> expand to (B, 3)
        eig_cond = target_eig.unsqueeze(0).expand(num_graphs, -1).to(device)
        
        # We only use 3 values (shape ratioes)
        c = eig_cond # (B, 3)
            
        # Overwrite the batch property so the model uses it for conditioning
        cfg_prop = getattr(lit_module.model, "cfg_property", "eigenvalues_normalized")
        setattr(batch, cfg_prop, c)
        
        # 3. Sample from model
        # Override sampling parameters
        lit_module.hparams.n_integration_steps = n_integration_steps
        # Force the model to use the specified guidance method
        lit_module.model.guidance_method = method
        if hasattr(lit_module, "model_ema") and lit_module.model_ema is not None:
            lit_module.model_ema.module.guidance_method = method

        if method == "CFG":
            lit_module.hparams.cfg_scale = scale
        elif method == "CG Self CFG-Trick":
            lit_module.hparams.cfg_scale = scale
            lit_module.model.guidance_lambda = lambda_scale_coef[0] if isinstance(lambda_scale_coef, (list, tuple)) else lambda_scale_coef
            lit_module.model.guidance_lambda_scale = lambda_scale_coef[1] if isinstance(lambda_scale_coef, (list, tuple)) and len(lambda_scale_coef) > 1 else 0.0
            lit_module.model.guidance_thinking_steps = thinking_steps
            if hasattr(lit_module, "model_ema") and lit_module.model_ema is not None:
                lit_module.model_ema.module.guidance_lambda = lit_module.model.guidance_lambda
                lit_module.model_ema.module.guidance_lambda_scale = lit_module.model.guidance_lambda_scale
                lit_module.model_ema.module.guidance_thinking_steps = thinking_steps
        else:
            lit_module.model.guidance_lambda = scale
            lit_module.model.guidance_lambda_scale = lambda_scale_coef
            lit_module.model.guidance_thinking_steps = thinking_steps
            if hasattr(lit_module, "model_ema") and lit_module.model_ema is not None:
                lit_module.model_ema.module.guidance_lambda = scale
                lit_module.model_ema.module.guidance_lambda_scale = lambda_scale_coef
                lit_module.model_ema.module.guidance_thinking_steps = thinking_steps
            lit_module.hparams.cfg_scale = 1.0 # default to 1.0 (no CFG effect) for other methods
        
        # Set a fixed seed for each batch to ensure the same initial noise across all methods and scales
        torch.manual_seed(42 + batch_idx)
            
        from src_gmmm.model.diffusion import compute_eigenvalues_differentiable

        with torch.no_grad():
            model = lit_module.get_model(ema=True)
            samples, traj = model.sample(
                batch,
                n_steps=n_integration_steps,
                epoch=lit_module.current_epoch,
                cfg_scale=lit_module.hparams.cfg_scale,
                return_traj=True
            )
            atoms_list = lit_module.atoms_from_tensors(**samples, ptr=batch.ptr)
            
            # Compute and store trajectory residuals
            traj_dir = os.path.join(plot_dir, "trajectories")
            os.makedirs(traj_dir, exist_ok=True)
            traj_csv_path = os.path.join(traj_dir, f"{target_name}_{method}_scale{scale}.csv")
            
            write_header = (batch_idx == 0)
            with open(traj_csv_path, mode='a' if not write_header else 'w', newline='') as f:
                import csv
                tw = csv.writer(f)
                if write_header:
                    tw.writerow(["Datapoint_Idx", "Timestep", "Eig1_MSE", "Eig2_MSE", "Eig3_MSE"])
                
                c_3d_traj = eig_cond # (B, 3)
                for step_idx, step_preds_pos in enumerate(traj["preds_pos"]):
                    y_pred = compute_eigenvalues_differentiable(step_preds_pos, batch.batch)
                    y_pred_3d = y_pred[:, :3]
                    mse_3d = ((y_pred_3d - c_3d_traj) ** 2).cpu().numpy()
                    
                    for b_idx in range(num_graphs):
                        global_idx = batch_idx * dataloader.batch_size + b_idx
                        tw.writerow([global_idx, step_idx, mse_3d[b_idx, 0], mse_3d[b_idx, 1], mse_3d[b_idx, 2]])
            
        if not images_saved:
            num_to_save = min(9, len(atoms_list))
            if num_to_save > 0:
                from src_gmmm.utils.callback import make_atoms_grid
                
                fig = make_atoms_grid(atoms_list[:num_to_save])
                save_dir = os.path.join(plot_dir, "images", method, target_name)
                os.makedirs(save_dir, exist_ok=True)
                
                save_path = os.path.join(save_dir, f"scale_{scale}.png")
                fig.savefig(save_path, bbox_inches='tight')
                plt.close(fig)
                
            images_saved = True
            
        # 4. Measure stability via GeomDrugsMetrics
        metrics.update(atoms_list)
        
        # We need per-molecule valid/stable to filter MSE
        # `metrics.molecule_stable` stores integer 1 or 0 for each processed molecule in order
        stabilities = metrics.molecule_stable[-num_graphs:]
        validities = metrics.valid[-num_graphs:]
        
        all_stabilities.extend(stabilities)
        all_validities.extend(validities)
        
        # 5. Measure MSE of eigenvalues
        # To compute eigenvalues, we need positions as PyG batch
        from torch_geometric.data import Data
        data_list = []
        for i, atoms in enumerate(atoms_list):
            pos = torch.tensor(atoms.positions, dtype=torch.float32)
            data_list.append(Data(pos=pos))
            
        generated_features = compute_eigenvalues_batched(data_list).to(device) # Returns (B, 4)
        
        # We compute squared error against 3 features
        generated_features_3d = generated_features[:, :3]
        c_3d = eig_cond # (B, 3)
        mses_3d = (generated_features_3d - c_3d) ** 2 # (B, 3)
        
        for i in range(len(stabilities)):
            all_mses_all_3d.append(mses_3d[i].cpu().numpy())
            if stabilities[i] == 1:
                all_mses_stable_3d.append(mses_3d[i].cpu().numpy())
            if validities[i] == 1:
                all_mses_valid_3d.append(mses_3d[i].cpu().numpy())
                
    mean_mse_stable_3d = np.mean(all_mses_stable_3d, axis=0) if len(all_mses_stable_3d) > 0 else np.full(3, float('nan'))
    mean_mse_valid_3d = np.mean(all_mses_valid_3d, axis=0) if len(all_mses_valid_3d) > 0 else np.full(3, float('nan'))
    mean_mse_all_3d = np.mean(all_mses_all_3d, axis=0) if len(all_mses_all_3d) > 0 else np.full(3, float('nan'))
    mean_stability = np.mean(all_stabilities) if len(all_stabilities) > 0 else 0.0
    mean_validity = np.mean(all_validities) if len(all_validities) > 0 else 0.0
    
    return mean_mse_stable_3d, mean_mse_valid_3d, mean_mse_all_3d, mean_stability, mean_validity


def parse_args():
    import sys
    import ast
    
    args = {
        "model_path": "",
        "num_datapoints": "all",
        "batch_size": 64,
        "cfg_scales": [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0],
        "lambda_cg_standard": [[0.0, 0.0], [0.01, 0.01], [0.05, 0.05], [0.1, 0.1], [0.5, 0.5]],
        "lambda_cg_self_cond": [[0.0, 0.0], [0.01, 0.01], [0.05, 0.05], [0.1, 0.1], [0.5, 0.5]],
        "thinking_steps": 1,
        "methods": ["CFG", "CG Std", "CG Self"],
        "plot_dir": "plot"
    }
    
    for arg in sys.argv[1:]:
        if "=" in arg:
            k, v = arg.split("=", 1)
            # Remove leading -- if user happens to use them
            k = k.lstrip("-")
            
            if k in args:
                if isinstance(args[k], bool):
                    args[k] = v.lower() in ("true", "1", "yes")
                elif isinstance(args[k], int):
                    args[k] = int(v)
                elif isinstance(args[k], float):
                    args[k] = float(v)
                elif isinstance(args[k], list):
                    args[k] = ast.literal_eval(v)
                else:
                    args[k] = v
                    
    return args

def get_pareto_frontier_max_min(Xs, Ys):
    points = sorted(zip(Xs, Ys), key=lambda p: (-p[0], p[1]))
    pareto_x, pareto_y = [], []
    min_y = float('inf')
    for x, y in points:
        if y < min_y:
            pareto_x.append(x)
            pareto_y.append(y)
            min_y = y
    return pareto_x[::-1], pareto_y[::-1]

def main():
    # Set global random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    args = parse_args()
    
    model_path = args["model_path"]
    num_datapoints = args["num_datapoints"]
    batch_size = args["batch_size"]
    cfg_scales = args["cfg_scales"]
    lambda_cg_standard = args["lambda_cg_standard"]
    lambda_cg_self_cond = args["lambda_cg_self_cond"]
    thinking_steps = args["thinking_steps"]
    methods = args["methods"]
    plot_dir = args["plot_dir"]
    
    os.makedirs(plot_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Save experiment details
    experiment_details = args
    with open(os.path.join(plot_dir, "experiment_details.json"), "w") as f:
        import json
        json.dump(experiment_details, f, indent=4)
        
    # 1. Load trained model
    print(f"Loading model from {model_path}...")
    job_dir, cfg = utils.load_cfg(checkpoint=model_path)
    lit_module: LitModule = hydra.utils.instantiate(cfg.lit_module)
    ckpt = torch.load(model_path, map_location=device)
    lit_module.load_state_dict(ckpt["state_dict"], strict=False)
    lit_module.to(device)
    lit_module.eval()
    
    # Infer configurations from cfg
    dataset_path = str(cfg.datamodule.val_path).replace("val.pt", "test.pt")
    json_infos_path = str(cfg.infos_path)
    n_integration_steps = cfg.lit_module.n_integration_steps
    predict_scale = (cfg.get("condition_dim", 3) == 4)
    
    # 2. Load dataset
    print(f"Loading dataset from {dataset_path}...")
    transform = hydra.utils.instantiate(cfg.datamodule.transform)
            
    test_data = Dataset(dataset_path, transform=transform)
    
    if num_datapoints != "all":
        num_dp = min(int(num_datapoints), len(test_data))
        indices = random.sample(range(len(test_data)), num_dp)
        subset_data = [test_data[i] for i in indices]
    else:
        subset_data = [test_data[i] for i in range(len(test_data))]
        
    dataloader = DataLoader(subset_data, batch_size=batch_size, shuffle=False)
    
    results = {
        method: {
            target: {
                "mses_stable": [],
                "mses_valid": [],
                "mses_all": [],
                "mses_3d_all": [],
                "mses_3d_stable": [],
                "stabilities": [],
                "validities": [],
                "scales": [],
                "scale_coefs": []
            } for target in TARGET_SHAPES
        } for method in methods
    }
    
    # Setup CSV Writer
    import csv
    csv_path = os.path.join(plot_dir, "evaluation_results.csv")
    csv_file = open(csv_path, mode='w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "Target", "Method", "Scale", "Scale_Coef", 
        "MSE_Stable", "MSE_Valid", "MSE_All", 
        "MSE_Eig1_All", "MSE_Eig2_All", "MSE_Eig3_All",
        "MSE_Eig1_Stable", "MSE_Eig2_Stable", "MSE_Eig3_Stable",
        "MSE_Eig1_Valid", "MSE_Eig2_Valid", "MSE_Eig3_Valid",
        "Stability", "Validity"
    ])
    
    # 3. Evaluate
    for target_name, target_eig in TARGET_SHAPES.items():
        print(f"--- Evaluating Target Shape: {target_name} ---")
        for method in methods:
            if method == "CFG":
                scales_to_test = [(s, 1.0) for s in cfg_scales]
            elif method == "CG Std":
                scales_to_test = lambda_cg_standard
            elif method == "CG Self":
                scales_to_test = lambda_cg_self_cond
            elif method == "CG Self CFG-Trick":
                scales_to_test = [(cfg_scales[i], lambda_cg_self_cond[i]) for i in range(len(cfg_scales))]
            else:
                scales_to_test = lambda_cg_standard
                
            for scale, scale_coef in scales_to_test:
                mse_stable_3d, mse_valid_3d, mse_all_3d, stab, valid = evaluate_shape_condition(
                    lit_module=lit_module,
                    dataloader=dataloader,
                    target_name=target_name,
                    target_eig=target_eig,
                    scale=scale,
                    method=method,
                    device=device,
                    json_infos_path=json_infos_path,
                    predict_scale=predict_scale,
                    n_integration_steps=n_integration_steps,
                    plot_dir=plot_dir,
                    lambda_scale_coef=scale_coef,
                    thinking_steps=thinking_steps
                )
                
                mse_stable = mse_stable_3d.mean()
                mse_valid = mse_valid_3d.mean()
                mse_all = mse_all_3d.mean()
                
                print(f"Method: {method}, Scale: {scale} (Coef: {scale_coef}) -> MSE (Stable): {mse_stable:.4f}, MSE (Valid): {mse_valid:.4f}, MSE (All): {mse_all:.4f}, Stability: {stab:.4f}, Validity: {valid:.4f}")
                
                results[method][target_name]["mses_stable"].append(mse_stable)
                results[method][target_name]["mses_valid"].append(mse_valid)
                results[method][target_name]["mses_all"].append(mse_all)
                results[method][target_name]["mses_3d_all"].append(mse_all_3d)
                results[method][target_name]["mses_3d_stable"].append(mse_stable_3d)
                results[method][target_name]["stabilities"].append(stab)
                results[method][target_name]["validities"].append(valid)
                results[method][target_name]["scales"].append(scale)
                results[method][target_name]["scale_coefs"].append(scale_coef)
                
                csv_writer.writerow([
                    target_name, method, scale, scale_coef, 
                    mse_stable, mse_valid, mse_all, 
                    mse_all_3d[0], mse_all_3d[1], mse_all_3d[2],
                    mse_stable_3d[0], mse_stable_3d[1], mse_stable_3d[2],
                    mse_valid_3d[0], mse_valid_3d[1], mse_valid_3d[2],
                    stab, valid
                ])
                csv_file.flush()
                
    csv_file.close()
                
    # 4. Plotting
    print(f"Generating plots into {plot_dir}/...")
    for target_name in TARGET_SHAPES:
        x_indices = np.arange(len(cfg_scales))
        
        def format_scale(val):
            try:
                val = float(val)
            except:
                return str(val)
            if val == 0: return "0"
            return f"{val:g}"

        x_tick_labels = []
        for i in range(len(cfg_scales)):
            cfg_val = format_scale(cfg_scales[i])
            if i == 0:
                lbl = f"CFG: {cfg_val}"
            else:
                lbl = f"{cfg_val}"
                
            if i < len(lambda_cg_standard):
                val = lambda_cg_standard[i]
                v = val[0] if isinstance(val, (list, tuple)) else val
                fmt_v = format_scale(v)
                if i == 0:
                    lbl += f"\nCG Std: {fmt_v}"
                else:
                    lbl += f"\n{fmt_v}"
                    
            if i < len(lambda_cg_self_cond):
                val = lambda_cg_self_cond[i]
                v = val[0] if isinstance(val, (list, tuple)) else val
                fmt_v = format_scale(v)
                if i == 0:
                    lbl += f"\nCG Self: {fmt_v}"
                else:
                    lbl += f"\n{fmt_v}"
                    
            x_tick_labels.append(lbl)
        
        def plot_mse(mse_key, y_label, title, filename_suffix):
            fig, ax1 = plt.subplots(figsize=(8, 6))
            for method in methods:
                mses = results[method][target_name][mse_key]
                mses = [0.0 if np.isnan(m) else m for m in mses]
                ax1.plot(x_indices[:len(mses)], mses, marker='o', label=method)
                    
            ax1.set_xticks(x_indices)
            ax1.set_xticklabels(x_tick_labels, fontsize=8)
            ax1.set_xlabel("Scale Values")
            ax1.set_ylabel(y_label)
            ax1.set_ylim(bottom=0)
            ax1.legend()
            ax1.grid(True)
            
            fig.savefig(os.path.join(plot_dir, f"{target_name.lower()}_{filename_suffix}.png"))
            plt.close(fig)
            
        plot_mse("mses_stable", "MSE of Eigenvalues (Stable only)", f"{target_name} Target - Eigenvalue MSE (Stable Only)", "mse_stable")
        
        # Plot 2: Stability
        fig, ax1 = plt.subplots(figsize=(8, 6))
        
        for method in methods:
            stabs = results[method][target_name]["stabilities"]
            ax1.plot(x_indices[:len(stabs)], stabs, marker='o', label=method)
            
        ax1.set_xticks(x_indices)
        ax1.set_xticklabels(x_tick_labels, fontsize=8)
        ax1.set_xlabel("Scale Values")
        ax1.set_ylabel("Molecular Stability")
        ax1.set_ylim(bottom=0)
        ax1.legend()
        ax1.grid(True)
        
        fig.savefig(os.path.join(plot_dir, f"{target_name.lower()}_stability.png"))
        plt.close(fig)
        
        # Plot 3: Validity
        fig, ax1 = plt.subplots(figsize=(8, 6))
        
        for method in methods:
            valids = results[method][target_name]["validities"]
            ax1.plot(x_indices[:len(valids)], valids, marker='o', label=method)
            
        ax1.set_xticks(x_indices)
        ax1.set_xticklabels(x_tick_labels, fontsize=8)
        ax1.set_xlabel("Scale Values")
        ax1.set_ylabel("Validity")
        ax1.set_ylim(bottom=0)
        ax1.legend()
        ax1.grid(True)
        
        fig.savefig(os.path.join(plot_dir, f"{target_name.lower()}_validity.png"))
        plt.close(fig)
        
        # Pareto Plot: Molecular Stability vs MSE Stable (Global Scatter Only)
        fig, ax = plt.subplots(figsize=(8, 6))
        all_stabs, all_mses_stable = [], []
        
        for method in methods:
            stabs = results[method][target_name]["stabilities"]
            mses_stable = [0.0 if np.isnan(m) else m for m in results[method][target_name]["mses_stable"]]
            
            ax.scatter(stabs, mses_stable, label=method, alpha=0.7)
            all_stabs.extend(stabs)
            all_mses_stable.extend(mses_stable)

        ax.set_xlabel("Molecular Stability")
        ax.set_ylabel("MSE (Stable)")
        ax.legend()
        ax.grid(True)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(bottom=0.0)
        fig.savefig(os.path.join(plot_dir, f"{target_name.lower()}_pareto_stability_vs_mse_stable.png"))
        plt.close(fig)
        
        # Pareto Plot: Molecular Stability vs MSE Stable (Per-Method)
        fig, ax = plt.subplots(figsize=(8, 6))
        global_max_y = max(all_mses_stable) if all_mses_stable else 1.0
        
        for method in methods:
            stabs = results[method][target_name]["stabilities"]
            mses_stable = [0.0 if np.isnan(m) else m for m in results[method][target_name]["mses_stable"]]
            
            # Use same color for scatter, step, and fill_between
            scatter = ax.scatter(stabs, mses_stable, label=method, alpha=0.7)
            color = scatter.get_facecolor()[0]
            
            if stabs and mses_stable:
                p_x, p_y = get_pareto_frontier_max_min(stabs, mses_stable)
                
                # Extend the frontier to the edges to make shading clean
                p_x_ext = [0.0] + p_x + [1.0]
                p_y_ext = [p_y[0]] + p_y + [p_y[-1]]
                
                ax.step(p_x_ext, p_y_ext, where='pre', color=color, linestyle='-', linewidth=1.5)
                ax.fill_between(p_x_ext, p_y_ext, global_max_y * 1.1, step='pre', color=color, alpha=0.1)

        ax.set_xlabel("Molecular Stability")
        ax.set_ylabel("MSE (Stable)")
        ax.legend()
        ax.grid(True)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(bottom=0.0, top=global_max_y * 1.1)
        fig.savefig(os.path.join(plot_dir, f"{target_name.lower()}_pareto_stability_vs_mse_stable_per_method.png"))
        plt.close(fig)
        
        # Heatmaps for Absolute Residuals
        heatmap_dir = os.path.join(plot_dir, "heatmaps")
        os.makedirs(heatmap_dir, exist_ok=True)
        
        y_labels = []
        heatmap_data = []
        cg_methods = ["CG Std", "CG Self"]
        for method in cg_methods:
            if method not in methods: continue
            
            scales = results[method][target_name]["scales"]
            scale_coefs = results[method][target_name]["scale_coefs"]
            mses_3d_stable = results[method][target_name]["mses_3d_stable"]
            
            for idx in range(len(scales)):
                sc = scales[idx]
                sc_coef = scale_coefs[idx]
                mse_3d = mses_3d_stable[idx]
                
                val_eig1 = mse_3d[0]
                val_eig2 = mse_3d[1]
                val_eig3 = mse_3d[2]
                
                heatmap_data.append([val_eig1, val_eig2, val_eig3])
                if method == "CG Std":
                    method_short = "Std"
                elif method == "CG Self":
                    method_short = "Self"
                else:
                    method_short = "Self CFG-Trick"
                
                if method == "CG Self CFG-Trick":
                    y_labels.append(f"{method_short} (CFG={format_scale(sc)}, Self={format_scale(sc_coef[0] if isinstance(sc_coef, (list,tuple)) else sc_coef)})")
                else:
                    y_labels.append(f"{method_short} (scale={format_scale(sc)})")
                
        if len(heatmap_data) > 0:
            heatmap_data = np.array(heatmap_data)
            
            # Save the numerical results
            csv_heatmap_path = os.path.join(heatmap_dir, f"{target_name.lower()}_heatmap_data.csv")
            with open(csv_heatmap_path, mode='w', newline='') as f:
                hw = csv.writer(f)
                hw.writerow(["Method_Scale", "Eig1_MSE", "Eig2_MSE", "Eig3_MSE"])
                for y_label, row in zip(y_labels, heatmap_data):
                    hw.writerow([y_label, row[0], row[1], row[2]])
            
            if np.isnan(heatmap_data).all():
                print(f"Skipping heatmap plots for {target_name} because all residuals are NaN.")
                continue
                
            import matplotlib.colors as mcolors
            
            # Non-log heatmap
            fig, ax = plt.subplots(figsize=(8, 10))
            cax = ax.imshow(heatmap_data, cmap='viridis', aspect='auto')
            ax.set_xticks(np.arange(3))
            ax.set_xticklabels(["Eig1", "Eig2", "Eig3"])
            ax.set_yticks(np.arange(len(y_labels)))
            ax.set_yticklabels(y_labels)
            fig.colorbar(cax, ax=ax, label="MSE")
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
            fig.tight_layout()
            fig.savefig(os.path.join(heatmap_dir, f"{target_name.lower()}_heatmap_linear.png"))
            plt.close(fig)
            
            # Log-scaled heatmap
            fig, ax = plt.subplots(figsize=(8, 10))
            safe_data = np.clip(heatmap_data, a_min=1e-10, a_max=None)
            
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                vmin = np.nanmin(safe_data)
                vmax = np.nanmax(safe_data)
                
            if np.isnan(vmin) or np.isnan(vmax):
                vmin, vmax = 1e-10, 1e-9
            elif vmin >= vmax:
                vmax = vmin * 10.0 if vmin > 0 else 1e-9
                
            cax = ax.imshow(safe_data, cmap='viridis', aspect='auto', norm=mcolors.LogNorm(vmin=vmin, vmax=vmax))
            ax.set_xticks(np.arange(3))
            ax.set_xticklabels(["Eig1", "Eig2", "Eig3"])
            ax.set_yticks(np.arange(len(y_labels)))
            ax.set_yticklabels(y_labels)
            fig.colorbar(cax, ax=ax, label="Log(MSE)")
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
            fig.tight_layout()
            fig.savefig(os.path.join(heatmap_dir, f"{target_name.lower()}_heatmap_log.png"))
            plt.close(fig)

    print("Evaluation complete!")

if __name__ == "__main__":
    main()
