from typing import Literal, Optional, Union, Dict
import warnings

import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data
from torch_geometric.utils import to_dense_batch

from ..model.continuous import ContinuousDiffusion
from ..model.score import EquivariantParameterization

class DistogramLoss(nn.Module):
    def __init__(self, num_bins: int = 128, cutoff: float = 3.2):
        super().__init__()
        self.num_bins = num_bins
        self.cutoff = cutoff

    def forward(self, preds_dist_edges: torch.Tensor, edge_node_index: torch.Tensor, pos: torch.Tensor):
        # preds_dist_edges: [num_edges, num_bins]
        # edge_node_index: [2, num_edges]
        # pos: [num_nodes, 3] (true positions)
        
        u, v = edge_node_index
        
        # 1. Compute exact distances for these edges
        dists = torch.norm(pos[u] - pos[v], dim=-1)
        
        # 2. Bucketize distances into bins
        bin_width = self.cutoff / (self.num_bins - 1)
        target_bins = torch.floor(dists / bin_width).long()
        # Anything > cutoff goes to the last bin
        target_bins = torch.clamp(target_bins, max=self.num_bins - 1)
        
        # 3. Compute combined loss
        loss = torch.nn.functional.cross_entropy(preds_dist_edges, target_bins, reduction='mean')
        
        return loss

class EquivariantDiffusion(nn.Module):
    def __init__(
        self,
        parameterization: EquivariantParameterization,
        diffusion_pos: Optional[ContinuousDiffusion],
        diffusion_h: Optional[ContinuousDiffusion],
        self_conditioning: bool = False, #self-conditioning: whether to use self-conditioning
        scprop: float = 0.9,  #self-conditioning: probability of using self-conditioning
        latent_recursion: bool = False, #latent recursion: whether to use latent recursion
        n: int = 1, #latent recursion: number of latent recursion steps
        K: int = 2, #latent recursion: number of deep recursion steps
        cfg: bool = False, # classifier-free guidance enabled
        cfg_prop: float = 0.5, # probability of replacing condition with null
        cfg_property: str = "eigenvalues_normalized", # property to condition on
        use_distogram: bool = False, # Whether to predict and compute distogram loss
        distogram_bins: int = 65,
        distogram_cutoff: float = 5.12,
    ):
        super().__init__()

        self.parameterization = parameterization
        self.diffusions = nn.ModuleDict({"pos": diffusion_pos, "h": diffusion_h})

        self.self_conditioning = self_conditioning #self-conditioning: whether to use self-conditioning
        self.scprop = scprop #self-conditioning: probability of using self-conditioning

        self.latent_recursion = latent_recursion
        self.n = n
        self.K = K 

        self.cfg = cfg
        self.cfg_prop = cfg_prop
        self.cfg_property = cfg_property
        self.use_distogram = use_distogram
        self.distogram_loss_fn = DistogramLoss(num_bins=distogram_bins, cutoff=distogram_cutoff)

    def loss_diffusion(self, t: torch.Tensor, batch: Batch | Data):
        latents, targets = self.training_targets(t=t, batch=batch)

        c = None
        if self.cfg:
            c = getattr(batch, self.cfg_property, None)
            if c is not None:
                c = c.view(batch.num_graphs, -1)
                if self.training:
                    # Standard 50/50% conditional/unconditional training
                    # Replace with zero vector (unconditional) with probability cfg_prop
                    mask = (torch.rand(c.size(0), 1, device=c.device) > self.cfg_prop).float()
                    c = c * mask
        
        #self-conditioning: run an inference step without backprop to get previous predictions
        prev_preds = None 
        if self.self_conditioning and not self.latent_recursion and torch.rand(1) < self.scprop:
            with torch.no_grad():
                prev_preds = self.parameterization.forward(
                    t=t,
                    **latents,
                    node_index=batch.batch,
                    edge_node_index=batch.edge_node_index,
                    c=c,
                )
        
        #latent recursion: arbitrary number of recursion steps

        z_prev = None
        if self.latent_recursion:
            skip = torch.rand(1) > self.scprop
            if not skip: #cold start for BOTH self-conditioning and latent recursion
                with torch.no_grad():
                    #deep recursion
                    for k in range(self.K-1):
                        #latent recursion
                        for i in range(self.n):
                            preds = self.parameterization.forward(
                                t=t,
                                **latents,
                                node_index=batch.batch,
                                edge_node_index=batch.edge_node_index,
                                prev_preds=prev_preds, #self-conditioning: pass previous predictions to the model
                                z_prev=z_prev, #latent recursion: pass previous latent states to the model
                                c=c,
                            )
                            z_prev = preds.get("z", None)
                        
                        # Update X1 (prevpreds) and Z, detach
                        preds = self.parameterization.forward(
                            t=t, **latents, node_index=batch.batch, edge_node_index=batch.edge_node_index,
                            prev_preds=prev_preds, z_prev=z_prev, c=c
                        )
                        z_prev = preds.get("z", None)
                        if not z_prev is None:
                            if isinstance(z_prev, list):
                                z_prev = [z.detach() for z in z_prev]
                            else:
                                z_prev = z_prev.detach()

                        if self.self_conditioning:
                            prev_preds = {k: v.detach() for k, v in preds.items() if k != "z"} #just copy

                    
                        
            # Run final iteration with tracking
            for i in range(self.n):
                #if skip:
                #    continue
                preds = self.parameterization.forward(
                    t=t, **latents, node_index=batch.batch, edge_node_index=batch.edge_node_index,
                    prev_preds=prev_preds, z_prev=z_prev, c=c
                )
                z_prev = preds.get("z", None)
            
            preds = self.parameterization.forward(
                t=t, **latents, node_index=batch.batch, edge_node_index=batch.edge_node_index,
                prev_preds=prev_preds, z_prev=z_prev, c=c
            )



        else:
            preds = self.parameterization.forward(
                t=t,
                **latents,
                node_index=batch.batch,
                edge_node_index=batch.edge_node_index,
                prev_preds=prev_preds, #self-conditioning: pass previous predictions to the model
                z_prev=z_prev, #latent recursion: pass previous latent states to the model
                c=c,
            )

        losses = {}

        for key in targets:
            loss = self.diffusions[key].loss_diffusion(
                preds[key],
                targets[key],
                t[batch.batch],  # cast time, when computing node-level property
                latents[key],
            )
            losses[key] = loss

        if "dist" in preds:
            losses["dist"] = self.distogram_loss_fn(
                preds_dist_edges=preds["dist"],
                edge_node_index=batch.edge_node_index,
                pos=batch.pos
            )

        return losses

    def training_targets(
        self, t: torch.Tensor, batch: Batch | Data
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        targets = {}

        index = batch.batch

        # position part
        if self.diffusion_pos is None:
            pos_t = batch.pos
        else:
            pos_t, target_pos_t = self.diffusion_pos.training_targets(
                t[index], batch.pos, index=batch.batch
            )
            targets["pos"] = target_pos_t

        # atomic species part
        if self.diffusion_h is None:
            h_t = batch.h
        else:
            h_t, target_h_t = self.diffusion_h.training_targets(
                t=t[index], x=batch.h, index=batch.batch
            )
            targets["h"] = target_h_t

        latents = {"pos": pos_t, "h": h_t}

        return latents, targets

    @torch.inference_mode()
    def sample_prior(
        self,
        batch: Batch | Data,
    ):
        index = batch.batch

        if self.diffusion_pos is None:
            assert batch.pos is not None
            pos = batch.pos
        else:
            pos = self.diffusion_pos.sample_prior(index)

        if self.diffusion_h is None:
            assert batch.h is not None
            h = batch.h
        else:
            num_nodes = len(index)
            h = self.diffusions["h"].sample_prior(n=num_nodes)

        return pos, h

    @torch.no_grad()
    def sample(
        self,
        batch: Batch | Data,
        method: Literal["em"] = "em",
        return_traj: bool = False,
        n_steps: int = 1000,
        ts: float = 1.0,
        tf: float = 1e-3,
        epoch: int = -1,
        cfg_scale: float = 1.0,
        c: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[
        dict[str, torch.Tensor],
        tuple[dict[str, torch.Tensor], dict[str, list[torch.Tensor]]],
    ]:

        node_index, edge_node_index = batch.batch, batch.edge_node_index
        num_graphs = batch.num_graphs
        device = node_index.device

        ts = torch.linspace(ts, tf, n_steps + 1, device=device)
        pos_t, h_t = self.sample_prior(batch=batch)

        if return_traj:
            traj = {
                "pos": [pos_t],
                "h": [h_t],
            }
        
        if self.cfg:
            if c is None:
                c = getattr(batch, self.cfg_property, None)
            if c is not None:
                c = c.view(num_graphs, -1)
            else:
                c = torch.zeros((num_graphs, 3), device=device)
        
        prev_preds = None  #self-conditioning: previous predictions
        for i in range(n_steps):
            t = ts[i]
            dt = ts[i + 1] - t

            t = torch.full((num_graphs, 1), t, device=device)

            #self-conditioning: update previous predictions
            if method == "em":
                pos_t, h_t, prev_preds = self.reverse_step_em( 
                    t=t,
                    dt=dt,
                    pos_t=pos_t,
                    h_t=h_t,
                    node_index=node_index,
                    edge_node_index=edge_node_index,
                    prev_preds=prev_preds, #self-conditioning: pass previous predictions to the model
                    c=c,
                    cfg_scale=cfg_scale,
                )
            if return_traj:
                traj["pos"].append(pos_t)
                traj["h"].append(h_t)

        samples = {
            "pos": pos_t,
            "h": h_t,
        }


        if return_traj:
            return samples, traj

        else:
            return samples

    def reverse_step_em(
        self,
        t: torch.Tensor,
        dt: torch.Tensor,
        pos_t: torch.Tensor,
        h_t: torch.Tensor,
        node_index: torch.Tensor,
        edge_node_index: torch.Tensor,
        prev_preds: Optional[dict[torch.Tensor, torch.Tensor]] = None, #self-conditioning: previous predictions
        c: Optional[torch.Tensor] = None,
        cfg_scale: float = 1.0,
    ):
        
        z_intermediates = []
        z_prev = prev_preds.get("z", None) if prev_preds is not None else None
        
        if self.latent_recursion:
            for i in range(self.n):
                preds = self.parameterization.forward(
                    t=t,
                    pos=pos_t,
                    h=h_t,
                    node_index=node_index,
                    edge_node_index=edge_node_index,
                    prev_preds=prev_preds, #self-conditioning: pass previous predictions to the model
                    z_prev=z_prev, #latent recursion: pass previous latent states to the model
                    c=c,
                )
                z_prev = preds.get("z", None)
        
        # get NN predictions
        if self.cfg and c is not None:
            c_null = torch.zeros_like(c)
            preds_uncond = self.parameterization.forward(
                t=t,
                pos=pos_t,
                h=h_t,
                node_index=node_index,
                edge_node_index=edge_node_index,
                prev_preds=prev_preds,
                z_prev=z_prev,
                c=c_null,
            )
            preds_cond = self.parameterization.forward(
                t=t,
                pos=pos_t,
                h=h_t,
                node_index=node_index,
                edge_node_index=edge_node_index,
                prev_preds=prev_preds,
                z_prev=z_prev,
                c=c,
            )
            preds = {}
            for key in preds_cond:
                if key == "z":
                    preds[key] = preds_cond[key]
                elif isinstance(preds_cond[key], torch.Tensor):
                    preds[key] = preds_uncond[key] + cfg_scale * (preds_cond[key] - preds_uncond[key])
                else:
                    preds[key] = preds_cond[key]
        else:
            preds = self.parameterization.forward(
                t=t,
                pos=pos_t,
                h=h_t,
                node_index=node_index,
                edge_node_index=edge_node_index,
                prev_preds=prev_preds, #self-conditioning: pass previous predictions to the model
                z_prev=z_prev, #latent recursion: pass previous latent states to the model
                c=c,
            )

        # reverse step on each modality
        if self.diffusion_pos:
            pos_t = self.diffusion_pos.reverse_step(
                t=t[node_index], x_t=pos_t, pred=preds["pos"], dt=dt, index=node_index
            )

        if self.diffusion_h:
            h_t = self.diffusion_h.reverse_step(
                t=t[node_index], x_t=h_t, pred=preds["h"], dt=dt, index=node_index
            )

        return pos_t, h_t, preds #self-conditioning: return predictions

    @property
    def diffusion_pos(self) -> Optional[ContinuousDiffusion]:
        return self.diffusions["pos"]

    @property
    def diffusion_h(self) -> Optional[ContinuousDiffusion]:
        return self.diffusions["h"]
