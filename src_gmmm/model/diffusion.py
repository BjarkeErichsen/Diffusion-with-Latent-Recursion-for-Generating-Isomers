from typing import Literal, Optional, Union, Dict
import warnings

import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data
from torch_geometric.utils import to_dense_batch
from scipy.optimize import linear_sum_assignment

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
        
        loss = torch.nn.functional.cross_entropy(preds_dist_edges, target_bins, reduction='mean')
        
        return loss

class MaskedHuberPairwiseLoss(nn.Module):
    def __init__(self, mask_cutoff: float = 2.0, delta: float = 1.0):
        super().__init__()
        self.mask_cutoff = mask_cutoff
        self.delta = delta

    def forward(self, preds_pos: torch.Tensor, edge_node_index: torch.Tensor, pos: torch.Tensor):
        u, v = edge_node_index
        
        # Calculate pairwise distance of true positions
        true_dists = torch.norm(pos[u] - pos[v], dim=-1)
        
        # Calculate pairwise distance of predicted positions
        pred_dists = torch.norm(preds_pos[u] - preds_pos[v], dim=-1)
        
        # Huber with mask, ignores every loss contribution from an edge whose true pairwise distance is greater than a threshold
        mask = (true_dists <= self.mask_cutoff).float()
        
        # Huber loss on the difference in pairwise distance
        loss = torch.nn.functional.huber_loss(pred_dists, true_dists, reduction='none', delta=self.delta)
        
        # Apply mask
        loss = (loss * mask).sum() / (mask.sum() + 1e-8)
        
        return loss

def compute_eigenvalues_differentiable(pos: torch.Tensor, batch_index: torch.Tensor) -> torch.Tensor:
    from torch_geometric.utils import to_dense_batch
    pos_dense, mask = to_dense_batch(pos, batch_index)
    
    num_nodes = mask.sum(dim=1, keepdim=True).unsqueeze(-1)
    mean_pos = (pos_dense * mask.unsqueeze(-1)).sum(dim=1, keepdim=True) / num_nodes
    
    pos_centered = (pos_dense - mean_pos) * mask.unsqueeze(-1)
    covs = torch.bmm(pos_centered.transpose(1, 2), pos_centered)
    
    e_unnorm = torch.linalg.eigh(covs).eigenvalues
    e_unnorm = torch.sort(e_unnorm, descending=True, dim=-1).values
    
    e_unnorm = torch.nn.functional.relu(e_unnorm)
    E = torch.sqrt(e_unnorm + 1e-8)
    S = E.sum(dim=-1, keepdim=True)
    S_safe = torch.clamp(S, min=1e-8)
    s_hat = E / S_safe
    
    return torch.cat([s_hat, S], dim=-1)

def grouped_sinkhorn_mse(preds_pos, targets_pos, atom_types, batch_indices, epsilon=0.0033, n_iters=5):
    """
    Computes permutation-invariant MSE loss per molecule and per atom type.
    
    Dimensionalities:
    N: Total number of atoms in the entire batch
    G: Number of unique (molecule, atom_type) groups
    K: Max number of atoms of the same type in any single molecule
    """
    from torch_geometric.utils import to_dense_batch
    
    # 1. Create a composite key for (batch_idx, atom_type)
    # mapped_types in R^N
    unique_types, mapped_types = torch.unique(atom_types, return_inverse=True)
    num_types = unique_types.size(0)
    group_idx = batch_indices * num_types + mapped_types 
    
    # 2. Dense batching based strictly on the composite group
    # pred_dense in R^{G x K x 3}, mask in R^{G x K}
    pred_dense, mask = to_dense_batch(preds_pos, group_idx)
    true_dense, _ = to_dense_batch(targets_pos, group_idx)
    
    # 3. Pairwise squared distance (MSE cost) isolated within each group
    # cost in R^{G x K x K}
    cost = torch.cdist(pred_dense, true_dense, p=2).pow(2)
    
    # 4. Valid pairs mask
    # valid_pairs in R^{G x K x K}. True only if BOTH i and j are real atoms.
    valid_pairs = mask.unsqueeze(2) & mask.unsqueeze(1)
    
    # Block invalid routings (dummy to dummy, real to dummy, etc.)
    cost_masked = cost.masked_fill(~valid_pairs, 1e9)
    
    # 5. Sinkhorn Initialization
    # K_dist in R^{G x K x K}
    K_dist = torch.exp(-cost_masked / epsilon)
    K_dist = K_dist * valid_pairs.float() # Strictly zero out padding
    
    # Marginals: Target sums (1 for real atoms, 0 for dummies)
    # a in R^{G x K x 1}, b in R^{G x 1 x K}
    a = mask.float().unsqueeze(2)
    b = mask.float().unsqueeze(1)
    
    # Scaling vectors
    u = torch.ones_like(a)
    v = torch.ones_like(b)
    
    # 6. Sinkhorn Iterations
    for _ in range(n_iters):
        # Update u: scale rows to match marginal 'a'
        u = a / (torch.matmul(K_dist, v.transpose(1, 2)) + 1e-8)
        # Update v: scale columns to match marginal 'b'
        v = b / (torch.matmul(u.transpose(1, 2), K_dist) + 1e-8)
        
    # 7. Final assignment matrix P in R^{G x K x K}
    P = u * K_dist * v.transpose(1, 2)
    
    # 8. Compute total loss and normalize by valid atoms to match standard MSE scale
    # We multiply the valid atom count by 3 since each position has 3 coordinates (X, Y, Z).
    # ACTUAL SINKHORN LOSS CALCULATION IS HERE
    total_loss = torch.sum(P * cost)
    mean_loss = total_loss / (mask.sum().float() * 3).clamp_min(1.0)
    
    return mean_loss


class EquivariantDiffusion(nn.Module):
    def __init__(
        self,
        parameterization: EquivariantParameterization,
        diffusion_pos: Optional[ContinuousDiffusion],
        diffusion_h: Optional[ContinuousDiffusion],
        self_conditioning: bool = False, #self-conditioning: whether to use self-conditioning
        scprop: float = 0.9,  #self-conditioning: probability of using self-conditioning
        latent_recursion: bool = False, #latent recursion: whether to use latent recursion
        latent_recursion_training_method: str = "standard", # latent recursion training method
        n: int = 1, #latent recursion: number of latent recursion steps
        K: int = 2, #latent recursion: number of deep recursion steps
        n_integration_steps: int = 250, # number of integration steps used in mimick inference training schedule
        train_cfg: bool = False, # classifier-free guidance enabled
        cfg: Optional[bool] = None, # backward compatibility for old configs
        guidance_method: str = "cfg",
        guidance_lambda: float = 0.01,
        guidance_lambda_scale: float = 0.01,
        guidance_thinking_steps: int = 1,
        cfg_prop: float = 0.5, # probability of replacing condition with null
        cfg_property: str = "eigenvalues_normalized", # property to condition on
        pair_dist_loss: str = "none", # Whether to predict and compute distogram loss
        distogram_bins: int = 65,
        distogram_cutoff: float = 5.12,
        distogram_mask_cutoff: float = 2.0,
        pair_dist_huber_delta: float = 1.0,
        permutation_invariant_loss: str = "none",
        condition_dim: int = 4,
        use_scale: bool = False,
    ):
        super().__init__()

        self.parameterization = parameterization
        self.diffusions = nn.ModuleDict({"pos": diffusion_pos, "h": diffusion_h}) 

        self.self_conditioning = self_conditioning #self-conditioning: whether to use self-conditioning
        self.scprop = scprop #self-conditioning: probability of using self-conditioning

        self.latent_recursion = latent_recursion
        self.latent_recursion_training_method = latent_recursion_training_method
        self.n = n
        self.K = K 
        self.n_integration_steps = n_integration_steps

        self.train_cfg = cfg if cfg is not None else train_cfg
        self.guidance_method = guidance_method
        self.guidance_lambda = guidance_lambda
        self.guidance_lambda_scale = guidance_lambda_scale
        self.guidance_thinking_steps = guidance_thinking_steps
        self.cfg_prop = cfg_prop
        self.cfg_property = cfg_property
        self.pair_dist_loss = pair_dist_loss
        self.permutation_invariant_loss = permutation_invariant_loss
        self.use_scale = use_scale
        if not self.use_scale:
            condition_dim = 3
        self.condition_dim = condition_dim
        self.c_null = nn.Parameter(torch.zeros(self.condition_dim))
        
        if self.pair_dist_loss == "huber_masked":
            self.pairwise_loss_fn = MaskedHuberPairwiseLoss(
                mask_cutoff=distogram_mask_cutoff,
                delta=pair_dist_huber_delta
            )
        elif self.pair_dist_loss != "none":
            raise NotImplementedError(f"Distogram loss variant {self.pair_dist_loss} not implemented yet.")
        else:
            self.pairwise_loss_fn = None

    def _compute_losses(
        self,
        preds: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
        latents: dict[str, torch.Tensor],
        t: torch.Tensor,
        batch: Batch | Data,
    ) -> dict[str, torch.Tensor]:
        losses = {}

        for key in targets:
            if key == "pos" and self.permutation_invariant_loss == "sinkhorn":
                loss = grouped_sinkhorn_mse(
                    preds_pos=preds["pos"],
                    targets_pos=targets["pos"],
                    atom_types=batch.h,
                    batch_indices=batch.batch
                )
            else:
                loss = self.diffusions[key].loss_diffusion(
                    preds[key],
                    targets[key],
                    t[batch.batch],  # cast time, when computing node-level property
                    latents[key],
                )
            losses[key] = loss

        if self.pair_dist_loss == "huber_masked" and "pos" in preds:
            losses["pairwise"] = self.pairwise_loss_fn(
                preds_pos=preds["pos"],
                edge_node_index=batch.edge_node_index,
                pos=batch.pos
            )

        return losses

    def loss_diffusion(self, t: torch.Tensor, batch: Batch | Data):
        latents, targets = self.training_targets(t=t, batch=batch)

        c = None
        if self.train_cfg:
            c = getattr(batch, self.cfg_property, None)
            if c is not None:
                c = c.view(batch.num_graphs, -1)
                if not self.use_scale and c.shape[-1] >= 4:
                    c = c[:, :3]
                if self.training:
                    # Standard 50/50% conditional/unconditional training
                    # Replace with learnable null vector (unconditional) with probability cfg_prop
                    mask = (torch.rand(c.size(0), 1, device=c.device) > self.cfg_prop).float()
                    c = c * mask + self.c_null.unsqueeze(0) * (1 - mask)
        
        #self-conditioning: run an inference step without backprop to get previous predictions
        prev_preds = None 
        if self.self_conditioning and torch.rand(1) < self.scprop:
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
            if torch.rand(1) < self.scprop: #cold start for BOTH self-conditioning and latent recursion
                with torch.no_grad():
                    #deep recursion
                    for k in range(self.K-1):
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

        # Execute training loop according to latent_recursion_training_method
        if not self.latent_recursion:
            preds = self.parameterization.forward(
                t=t,
                **latents,
                node_index=batch.batch,
                edge_node_index=batch.edge_node_index,
                prev_preds=prev_preds,
                z_prev=z_prev,
                c=c,
            )
            return self._compute_losses(preds, targets, latents, t, batch)

        elif self.latent_recursion_training_method == "standard":
            for i in range(self.n):
                preds = self.parameterization.forward(
                    t=t, **latents, node_index=batch.batch, edge_node_index=batch.edge_node_index,
                    prev_preds=prev_preds, z_prev=z_prev, c=c
                )
                z_prev = preds.get("z", None)
                if self.self_conditioning:
                    prev_preds = {k: v for k, v in preds.items() if k != "z"}

            preds = self.parameterization.forward(
                t=t, **latents, node_index=batch.batch, edge_node_index=batch.edge_node_index,
                prev_preds=prev_preds, z_prev=z_prev, c=c
            )
            return self._compute_losses(preds, targets, latents, t, batch)

        elif self.latent_recursion_training_method == "mimick_inference":
            dt = torch.full_like(t, -(1.0 - 1e-3) / self.n_integration_steps)
            current_t = t.clone()
            current_latents = {k: v.clone() for k, v in latents.items()}

            for i in range(self.n):
                preds = self.parameterization.forward(
                    t=current_t, **current_latents, node_index=batch.batch, edge_node_index=batch.edge_node_index,
                    prev_preds=prev_preds, z_prev=z_prev, c=c
                )
                z_prev = preds.get("z", None)
                prev_preds = {k: v for k, v in preds.items() if k != "z"}

                with torch.no_grad():
                    pos_integrated = self.diffusions["pos"].reverse_step(
                        t=current_t[batch.batch],
                        x_t=current_latents["pos"],
                        pred=preds["pos"],
                        dt=dt[batch.batch],
                        index=batch.batch,
                    )
                current_latents["pos"] = pos_integrated.detach()

                current_t = torch.clamp(current_t + dt, min=1e-3)

            preds = self.parameterization.forward(
                t=current_t, **current_latents, node_index=batch.batch, edge_node_index=batch.edge_node_index,
                prev_preds=prev_preds, z_prev=z_prev, c=c
            )
            return self._compute_losses(preds, targets, current_latents, current_t, batch)

        elif self.latent_recursion_training_method == "mimick_inference_bpt_integrator":
            dt = torch.full_like(t, -(1.0 - 1e-3) / self.n_integration_steps)
            current_t = t.clone()
            current_latents = {k: v.clone() for k, v in latents.items()}

            for i in range(self.n):
                preds = self.parameterization.forward(
                    t=current_t, **current_latents, node_index=batch.batch, edge_node_index=batch.edge_node_index,
                    prev_preds=prev_preds, z_prev=z_prev, c=c
                )
                z_prev = preds.get("z", None)
                prev_preds = {k: v for k, v in preds.items() if k != "z"}

                pos_integrated = self.diffusions["pos"].reverse_step(
                    t=current_t[batch.batch],
                    x_t=current_latents["pos"],
                    pred=preds["pos"],
                    dt=dt[batch.batch],
                    index=batch.batch,
                )
                current_latents["pos"] = pos_integrated

                current_t = torch.clamp(current_t + dt, min=1e-3)

            preds = self.parameterization.forward(
                t=current_t, **current_latents, node_index=batch.batch, edge_node_index=batch.edge_node_index,
                prev_preds=prev_preds, z_prev=z_prev, c=c
            )
            return self._compute_losses(preds, targets, current_latents, current_t, batch)

        elif self.latent_recursion_training_method == "mimick_inference_loss_at_each_timestep":
            dt = torch.full_like(t, -(1.0 - 1e-3) / self.n_integration_steps)
            current_t = t.clone()
            current_latents = {k: v.clone() for k, v in latents.items()}
            all_step_losses = []

            for i in range(self.n + 1):
                preds = self.parameterization.forward(
                    t=current_t, **current_latents, node_index=batch.batch, edge_node_index=batch.edge_node_index,
                    prev_preds=prev_preds, z_prev=z_prev, c=c
                )
                step_losses = self._compute_losses(preds, targets, current_latents, current_t, batch)
                all_step_losses.append(step_losses)

                if i < self.n:
                    z_prev = preds.get("z", None)
                    if z_prev is not None:
                        if isinstance(z_prev, list):
                            z_prev = [z.detach() for z in z_prev]
                        else:
                            z_prev = z_prev.detach()

                    prev_preds = {k: v.detach() for k, v in preds.items() if k != "z"}

                    with torch.no_grad():
                        pos_integrated = self.diffusions["pos"].reverse_step(
                            t=current_t[batch.batch],
                            x_t=current_latents["pos"],
                            pred=preds["pos"],
                            dt=dt[batch.batch],
                            index=batch.batch,
                        )
                    current_latents["pos"] = pos_integrated.detach()

                    current_t = torch.clamp(current_t + dt, min=1e-3)

            losses = {}
            for key in all_step_losses[0]:
                losses[key] = torch.stack([step[key] for step in all_step_losses]).mean()
            return losses

        else:
            raise ValueError(f"Unknown latent_recursion_training_method: '{self.latent_recursion_training_method}'")

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

    @torch.inference_mode()
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
        
        if self.guidance_method == "none":
            c = self.c_null.unsqueeze(0).expand(num_graphs, -1) if self.train_cfg else None
        elif self.guidance_method in ["CFG", "CG Std", "CG Self", "CG Self CFG-Trick"]:
            if c is None:
                c = getattr(batch, self.cfg_property, None)
            if c is not None:
                c = c.view(num_graphs, -1)
                if not getattr(self, "use_scale", False) and c.shape[-1] >= 4:
                    c = c[:, :3]
            else:
                if not self.training:
                    raise ValueError(f"Guidance method {self.guidance_method} requires condition `c`, but it was not provided and `batch.{self.cfg_property}` was missing.")
                c = self.c_null.unsqueeze(0).expand(num_graphs, -1)
        
        prev_preds = None  #self-conditioning: previous predictions
        for i in range(n_steps):
            t = ts[i]
            dt = ts[i + 1] - t

            t = torch.full((num_graphs, 1), t, device=device)

            x_0_prev_original = prev_preds["pos"].clone().detach() if prev_preds is not None else None
            
            if self.guidance_method in ['CG Self', 'CG Self CFG-Trick'] and prev_preds is not None and c is not None:
                x_0_prev = prev_preds["pos"].clone().detach()
                for step in range(self.guidance_thinking_steps):
                    with torch.enable_grad():
                        x_0_prev = x_0_prev.detach().requires_grad_(True)
                        y_pred = compute_eigenvalues_differentiable(x_0_prev, node_index)
                        if getattr(self, "use_scale", False) and c.shape[-1] == 4 and y_pred.shape[-1] == 4:
                            loss = torch.mean((c[:, :3] - y_pred[:, :3])**2) + getattr(self, "guidance_lambda_scale", 0.01) * torch.mean((c[:, 3] - y_pred[:, 3])**2)
                        else:
                            if not getattr(self, "use_scale", False):
                                loss = torch.mean((c - y_pred[:, :3])**2)
                            else:
                                loss = torch.mean((c - y_pred)**2)
                        grad_x0 = torch.autograd.grad(loss, x_0_prev)[0]
                        x_0_prev = (x_0_prev - self.guidance_lambda * grad_x0).detach()
                    
                    if step < self.guidance_thinking_steps - 1:
                        temp_prev_preds = prev_preds.copy()
                        temp_prev_preds["pos"] = x_0_prev
                        
                        preds_temp = self.parameterization.forward(
                            t=t, pos=pos_t, h=h_t, node_index=node_index, 
                            edge_node_index=edge_node_index, prev_preds=temp_prev_preds, 
                            z_prev=temp_prev_preds.get("z", None), 
                            c=self.c_null.unsqueeze(0).expand(num_graphs, -1)
                        )
                        x_0_prev = preds_temp["pos"].clone().detach()
                        
                if self.guidance_method == 'CG Self':
                    prev_preds["pos"] = x_0_prev
                else:
                    prev_preds["pos_guided"] = x_0_prev
                    prev_preds["pos_original"] = x_0_prev_original
            
            elif self.guidance_method == 'CG Std' and c is not None:
                for _ in range(self.guidance_thinking_steps):
                    with torch.enable_grad():
                        pos_t_in = pos_t.clone().detach().requires_grad_(True)
                        preds_guidance = self.parameterization.forward(
                            t=t, pos=pos_t_in, h=h_t, node_index=node_index, 
                            edge_node_index=edge_node_index, prev_preds=prev_preds, z_prev=prev_preds.get("z", None) if prev_preds else None, c=self.c_null.unsqueeze(0).expand(num_graphs, -1)
                        )
                        y_pred = compute_eigenvalues_differentiable(preds_guidance["pos"], node_index)
                        if getattr(self, "use_scale", False) and c.shape[-1] == 4 and y_pred.shape[-1] == 4:
                            loss = torch.mean((c[:, :3] - y_pred[:, :3])**2) + getattr(self, "guidance_lambda_scale", 0.01) * torch.mean((c[:, 3] - y_pred[:, 3])**2)
                        else:
                            if not getattr(self, "use_scale", False):
                                loss = torch.mean((c - y_pred[:, :3])**2)
                            else:
                                loss = torch.mean((c - y_pred)**2)
                        grad_xt = torch.autograd.grad(loss, pos_t_in)[0]
                    pos_t = pos_t - self.guidance_lambda * grad_xt.detach()

            #self-conditioning: update previous predictions
            if method == "em":
                c_for_em = c if self.guidance_method in ["CFG", "none"] else self.c_null.unsqueeze(0).expand(num_graphs, -1)
                pos_t, h_t, prev_preds = self.reverse_step_em( 
                    t=t,
                    dt=dt,
                    pos_t=pos_t,
                    h_t=h_t,
                    node_index=node_index,
                    edge_node_index=edge_node_index,
                    prev_preds=prev_preds, #self-conditioning: pass previous predictions to the model
                    c=c_for_em,
                    cfg_scale=cfg_scale,
                )
            if return_traj:
                traj["pos"].append(pos_t)
                traj["h"].append(h_t)
                if "preds_pos" not in traj:
                    traj["preds_pos"] = []
                traj["preds_pos"].append(prev_preds["pos"])

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
            #for i in range(self.n):
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
            #z_prev = preds.get("z", None)
        
        
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

        # get NN predictions
        if self.guidance_method == "CFG" and c is not None:
            c_null = self.c_null.unsqueeze(0).expand(c.size(0), -1)
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
        elif self.guidance_method == "CG Self CFG-Trick" and prev_preds is not None and c is not None:
            prev_preds_guided = prev_preds.copy()
            prev_preds_guided["pos"] = prev_preds["pos_guided"]
            
            prev_preds_unguid = prev_preds.copy()
            prev_preds_unguid["pos"] = prev_preds["pos_original"]
            
            c_null = self.c_null.unsqueeze(0).expand(c.size(0), -1)
            
            preds_unguid = self.parameterization.forward(
                t=t,
                pos=pos_t,
                h=h_t,
                node_index=node_index,
                edge_node_index=edge_node_index,
                prev_preds=prev_preds_unguid,
                z_prev=z_prev,
                c=c_null,
            )
            
            preds_guided = self.parameterization.forward(
                t=t,
                pos=pos_t,
                h=h_t,
                node_index=node_index,
                edge_node_index=edge_node_index,
                prev_preds=prev_preds_guided,
                z_prev=z_prev,
                c=c_null,
            )
            
            preds = {}
            for key in preds_guided:
                if key == "z":
                    preds[key] = preds_guided[key]
                elif isinstance(preds_guided[key], torch.Tensor):
                    preds[key] = preds_unguid[key] + cfg_scale * (preds_guided[key] - preds_unguid[key])
                else:
                    preds[key] = preds_guided[key]

        # Variance Preserving (VP-SDE) Euler-Maruyama integration step on each modality
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
