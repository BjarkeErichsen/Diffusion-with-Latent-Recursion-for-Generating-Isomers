import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from src_gmmm.lit.module import LitModule
from src_gmmm.data.datamodule import DataModule
from hydra import compose, initialize

def main(ckpt_path: str, output_path: str):
    import os
    import hydra
    from omegaconf import OmegaConf
    
    # Path to the run's config
    run_dir = os.path.dirname(os.path.dirname(ckpt_path))
    cfg_path = os.path.join(run_dir, ".hydra", "config.yaml")
    print(f"Loading config from {cfg_path}...")
    cfg = OmegaConf.load(cfg_path)
        
    print("Instantiating model...")
    core_model = hydra.utils.instantiate(cfg.lit_module.model)
    
    print(f"Loading checkpoint {ckpt_path}...")
    # strict=False because some ema keys might be slightly different or missing
    model = LitModule.load_from_checkpoint(ckpt_path, model=core_model, strict=False, map_location="cpu")
    model.eval()
    
    print("Loading data...")
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.setup(stage="fit")
    
    val_loader = datamodule.val_dataloader()
    
    # Find a small molecule (e.g. between 5 and 15 atoms) to animate
    small_data = None
    for batch_idx, batch in enumerate(val_loader):
        for data in batch.to_data_list():
            if 5 <= data.num_nodes <= 15:
                small_data = data
                break
        if small_data is not None:
            break
            
    from torch_geometric.data import Batch
    batch = Batch.from_data_list([small_data])
    batch = batch.to('cpu')
    
    # We just want to denoise one molecule, let's take the first one from the batch
    # But wait, sample takes a batch. Let's pass the whole batch and then extract the first trajectory.
    print("Sampling trajectory...")
    n_steps = 250
    # Make sure we ask for trajectory
    res = model.model.sample(batch, return_traj=True, n_steps=n_steps)
    
    # res is a tuple: (final_dict, traj_dict)
    final_dict, traj_dict = res
    pos_traj = traj_dict["pos"] # List of (N_atoms, 3) across steps
    h_traj = traj_dict["h"]
    
    # Extract just the first molecule
    # batch.batch is the graph index
    mask = (batch.batch == 0)
    
    # Convert list of tensors to a single tensor (n_steps+1, N_atoms, 3)
    # Extract only nodes for the first molecule
    mol_pos_traj = [pos[mask].detach().cpu().numpy() for pos in pos_traj]
    mol_h_traj = [h[mask].detach().cpu().numpy() for h in h_traj]
    
    print(f"Animating {len(mol_pos_traj)} steps...")
    
    from ase.data.colors import jmol_colors
    atomic_numbers = [1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53, 83]
    atom_types_str = ["H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Br", "I", "Bi"]
    
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_axis_off()
    
    # Precompute limits
    all_pos = np.concatenate(mol_pos_traj, axis=0)
    min_b, max_b = all_pos.min(axis=0), all_pos.max(axis=0)
    
    # Scatter plot
    scat = ax.scatter([], [], [], s=100, c='blue', alpha=0.8, edgecolors='k')
    
    title = ax.set_title("Denoising Step 0")

    def update(frame):
        pos = mol_pos_traj[frame]
        
        # update scatter data
        # For 3D scatter, we need to set _offsets3d
        scat._offsets3d = (pos[:, 0], pos[:, 1], pos[:, 2])
        
        # Set colors based on atom types. Since diffusion_h is None, h is the atomic numbers directly!
        h = mol_h_traj[frame]
        atom_nums = np.atleast_1d(h).astype(int)
        
        # Safe mapping because sometimes num can be out of range
        def get_color(num):
            try:
                return jmol_colors[num]
            except IndexError:
                return (0.5, 0.5, 0.5) # Gray fallback
                
        colors = [get_color(num) for num in atom_nums]
        scat.set_color(colors)
        scat.set_edgecolor('k')
        
        # Update legend
        ax.legend_.remove() if ax.get_legend() else None
        unique_nums = np.unique(atom_nums)
        from matplotlib.lines import Line2D
        
        atom_num_to_str = {num: string for num, string in zip(atomic_numbers, atom_types_str)}
        
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label=atom_num_to_str.get(num, "Unknown"),
                   markerfacecolor=get_color(num), markersize=10, markeredgecolor='k')
            for num in unique_nums
        ]
        ax.legend(handles=legend_elements, loc='upper right', title="Atom Types")
        
        ax.set_xlim(min_b[0], max_b[0])
        ax.set_ylim(min_b[1], max_b[1])
        ax.set_zlim(min_b[2], max_b[2])
        
        # Reverse steps since it goes from T to 0
        title.set_text(f"Denoising Step {n_steps - frame}/{n_steps}")
        return scat, title
    
    ani = animation.FuncAnimation(fig, update, frames=len(mol_pos_traj), interval=50, blit=False)
    
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    print(f"Saving animation to {output_path}...")
    ani.save(output_path, writer='pillow', fps=20)
    print("Done generating animation!")
    
    # Save the true molecule visualization
    print("Saving true molecule visualization...")
    true_pos = small_data.pos.detach().cpu().numpy()
    true_h = np.atleast_1d(small_data.h.detach().cpu().numpy()).astype(int)
    
    def get_color(num):
        try:
            return jmol_colors[num]
        except IndexError:
            return (0.5, 0.5, 0.5) # Gray fallback
            
    fig_true = plt.figure(figsize=(8, 6))
    ax_true = fig_true.add_subplot(111, projection='3d')
    ax_true.set_axis_off()
    
    colors_true = [get_color(num) for num in true_h]
    scat_true = ax_true.scatter(true_pos[:, 0], true_pos[:, 1], true_pos[:, 2], s=100, c=colors_true, alpha=0.8, edgecolors='k')
    
    unique_nums_true = np.unique(true_h)
    from matplotlib.lines import Line2D
    
    atom_num_to_str = {num: string for num, string in zip(atomic_numbers, atom_types_str)}
    
    legend_elements_true = [
        Line2D([0], [0], marker='o', color='w', label=atom_num_to_str.get(num, "Unknown"),
               markerfacecolor=get_color(num), markersize=10, markeredgecolor='k')
        for num in unique_nums_true
    ]
    ax_true.legend(handles=legend_elements_true, loc='upper right', title="True Atom Types")
    ax_true.set_xlim(min_b[0], max_b[0])
    ax_true.set_ylim(min_b[1], max_b[1])
    ax_true.set_zlim(min_b[2], max_b[2])
    ax_true.set_title("True Molecule")
    
    true_out_path = os.path.join(out_dir, "true_molecule.png") if out_dir else "true_molecule.png"
    fig_true.savefig(true_out_path, dpi=300, bbox_inches='tight')
    plt.close(fig_true)
    print(f"Saved true molecule to {true_out_path}")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--output", type=str, default="evals/denoising_animation.gif")
    args = parser.parse_args()
    
    main(args.ckpt, args.output)
