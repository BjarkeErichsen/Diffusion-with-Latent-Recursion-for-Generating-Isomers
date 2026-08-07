import os
from pathlib import Path
from typing import Any, Optional, Union

import ase
import numpy as np
from ase.visualize.plot import plot_atoms
from matplotlib import pyplot as plt
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers.wandb import WandbLogger

from ..data.utils import save_images

def get_pairwise_distances(atoms_lst: list[ase.Atoms], sym1: str, sym2: str):
    dists = []
    for atoms in atoms_lst:
        symbols = np.array(atoms.get_chemical_symbols())
        idx1 = np.where(symbols == sym1)[0]
        idx2 = np.where(symbols == sym2)[0]
        if len(idx1) == 0 or len(idx2) == 0:
            continue
        dist_mat = atoms.get_all_distances()
        if sym1 == sym2:
            # upper triangle
            for i in range(len(idx1)):
                for j in range(i + 1, len(idx1)):
                    dists.append(dist_mat[idx1[i], idx1[j]])
        else:
            for i in idx1:
                for j in idx2:
                    dists.append(dist_mat[i, j])
    return np.array(dists)


def get_wandb_logger(trainer: Trainer) -> Optional[WandbLogger]:
    wandb_logger = None
    for logger in trainer.loggers:
        if isinstance(logger, WandbLogger):
            if wandb_logger is not None:
                raise ValueError(
                    "More than one WandbLogger was found in the list of loggers"
                )

            wandb_logger = logger

    return wandb_logger


def make_atoms_grid(atoms_lst: list[ase.Atoms]):
    bs = len(atoms_lst)
    nrows = ncols = int(np.ceil(np.sqrt(bs)))

    fig, _ = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 3, nrows * 3))
    for i, ax in enumerate(fig.axes):
        if i >= len(atoms_lst):
            break
        atoms = atoms_lst[i]
        plot_atoms(atoms, ax)
    return fig


class LogSampledAtomsCallback(Callback):
    def __init__(
        self,
        dirpath: Union[Path, str],
        save_atoms: bool = True,
        num_log_wandb: int = 25,
        prefix_with_epoch: bool = True,
        atom_type_pairwise_distogram_plots: bool = True,
    ):
        self.dirpath = dirpath
        self.save_atoms = save_atoms
        self.num_log_wandb = num_log_wandb
        self.prefix_with_epoch = prefix_with_epoch
        self.atom_type_pairwise_distogram_plots = atom_type_pairwise_distogram_plots

        self.atoms_lst: list[ase.Atoms] = ...
        self.atoms_gt_lst: list[ase.Atoms] = ...

    def on_validation_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        super().on_validation_start(trainer, pl_module)
        self.atoms_lst = []
        self.atoms_gt_lst = []

    def on_test_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        return self.on_validation_start(trainer, pl_module)

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: list[ase.Atoms],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        self.atoms_lst.extend(outputs)
        
        atoms_gt = pl_module.atoms_from_tensors(batch.h, batch.pos, batch.ptr)
        self.atoms_gt_lst.extend(atoms_gt)

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: list[ase.Atoms],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        return self.on_validation_batch_end(
            trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
        )

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:

        super().on_validation_epoch_end(trainer, pl_module)
        epoch = pl_module.current_epoch

        if self.prefix_with_epoch:
            dirpath = os.path.join(self.dirpath, str(epoch))
        else:
            dirpath = self.dirpath

        if not os.path.exists(dirpath):
            os.makedirs(dirpath, exist_ok=True)

        if self.save_atoms:
            save_path = os.path.join(dirpath, "samples.xyz")
            save_images(self.atoms_lst, filename=save_path)
            
            save_path_gt = os.path.join(dirpath, "samples_gt.xyz")
            save_images(self.atoms_gt_lst, filename=save_path_gt)

        if self.num_log_wandb:
            logger: WandbLogger = get_wandb_logger(trainer)

            idx = min(len(self.atoms_lst), self.num_log_wandb)
            fig = make_atoms_grid(self.atoms_lst[-idx:])
            
            idx_gt = min(len(self.atoms_gt_lst), self.num_log_wandb)
            fig_gt = make_atoms_grid(self.atoms_gt_lst[-idx_gt:])

            if logger is not None:
                logger.log_image(f"val/images", [fig])
                logger.log_image(f"val/images_gt", [fig_gt])
                
                if self.atom_type_pairwise_distogram_plots:
                    dist_fig, axs = plt.subplots(2, 2, figsize=(12, 10))
                    pairs = [("C", "C"), ("C", "H"), ("O", "H"), ("C", "N")]
                    
                    # 128 bins between 0 and 2.0
                    bins = np.linspace(0, 2.0, 128)
                    
                    for ax, (sym1, sym2) in zip(axs.flatten(), pairs):
                        dists_gt = get_pairwise_distances(self.atoms_gt_lst, sym1, sym2)
                        dists_pred = get_pairwise_distances(self.atoms_lst, sym1, sym2)
                        
                        ax.hist(dists_gt, bins=bins, alpha=0.5, label='True')
                        ax.hist(dists_pred, bins=bins, alpha=0.5, label='Predicted')
                        ax.set_title(f"{sym1}-{sym2} Distances")
                        ax.set_yscale('log')
                        ax.set_xlim(0, 2.0)
                        ax.set_xlabel("Distance (Å)")
                        ax.set_ylabel("Count (log)")
                        ax.legend()
                    
                    dist_fig.tight_layout()
                    logger.log_image(f"val/pairwise_distograms", [dist_fig])
                    plt.close(dist_fig)

            plt.close(fig)
            plt.close(fig_gt)

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        return self.on_validation_epoch_end(trainer, pl_module)
