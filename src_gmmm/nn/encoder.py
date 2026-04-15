from typing import Optional

import torch
import torch.nn as nn
from torch_scatter import scatter_sum

from ..nn.layers import EdgeEmbedding, EquivLayerNorm, FourierEmbedding
from ..nn.layers import SelfConditioningResidualLayer #self-conditioning
from ..nn.layers import LatentSyncModule #latent-recursion

class InteractionLayer(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
    ):
        super(InteractionLayer, self).__init__()
        self.node_dim = node_dim
        self.W = nn.Linear(edge_dim, 3 * node_dim)
        self.msg_nn = nn.Sequential(
            nn.Linear(node_dim, node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, 3 * node_dim),
        )
        self.edge_inference_nn = nn.Sequential(
            nn.Linear(node_dim, 1),
            nn.Sigmoid(),
        )

        self.ln = EquivLayerNorm(dims=(node_dim, node_dim))

    def forward(
        self,
        node_states_s: torch.Tensor,
        node_states_v: torch.Tensor,
        edge_states: torch.Tensor,
        unit_vectors: torch.Tensor,
        node_index: torch.Tensor,
        edge_node_index: torch.Tensor,
    ):
        src_idx, dst_idx = edge_node_index

        node_states_s, node_states_v = self.ln.forward(
            node_states_s, node_states_v, node_index
        )

        W = self.W(edge_states)
        phi = self.msg_nn(node_states_s)
        Wphi = W * phi[src_idx]  # num_edges, 3*node_size
        phi_s, phi_vv, phi_vs = torch.split(Wphi, self.node_dim, dim=1)
        edge = self.edge_inference_nn(phi_s)
        messages_s = phi_s * edge
        messages_v = (
            node_states_v[src_idx] * phi_vv[:, None, :]
            + phi_vs[:, None, :] * unit_vectors[..., None]
        ) * edge[..., None]

        reduced_messages_s = scatter_sum(
            messages_s, dst_idx, dim=0, out=torch.zeros_like(node_states_s)
        )
        reduced_messages_v = scatter_sum(
            messages_v, dst_idx, dim=0, out=torch.zeros_like(node_states_v)
        )

        return (
            node_states_s + reduced_messages_s,
            node_states_v + reduced_messages_v,
        )


class UpdateLayer(nn.Module):
    def __init__(
        self,
        node_dim: int,
    ):
        super(UpdateLayer, self).__init__()
        self.node_dim = node_dim
        self.UV = nn.Linear(node_dim, 2 * node_dim, bias=False)
        self.UV_nn = nn.Sequential(
            nn.Linear(2 * node_dim, node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, 3 * node_dim),
        )

    def forward(self, node_states_s: torch.Tensor, node_states_v: torch.Tensor):
        UVv = self.UV(node_states_v)  # (n_nodes, 3, 2 * F)
        Uv, Vv = torch.split(UVv, self.node_dim, -1)  # (n_nodes, 3, F)
        Vv_norm = torch.sqrt(
            torch.sum(Vv**2, dim=1) + 1e-6
        )  # norm over spatial components

        a = self.UV_nn(torch.cat((Vv_norm, node_states_s), dim=1))
        a_vv, a_sv, a_ss = torch.split(a, self.node_dim, dim=1)

        inner_prod = torch.sum(Uv * Vv, dim=1)
        delta_s = a_ss + a_sv * inner_prod
        delta_v = a_vv[:, None, :] * Uv  # a_vv.shape = (n_nodes, F)

        return node_states_s + delta_s, node_states_v + delta_v


class EdgeLayer(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, residual: bool = False):
        super().__init__()
        self.node_dim = node_dim
        self.edge_nn = nn.Sequential(
            nn.Linear(edge_dim + 2 * node_dim, 2 * node_dim),
            nn.SiLU(),
            nn.Linear(2 * node_dim, edge_dim),
        )
        self.residual = residual
        self.mask = nn.Parameter(
            torch.as_tensor([1.0 for _ in range(edge_dim)]), requires_grad=True
        )

    def forward(
        self,
        node_states: torch.Tensor,
        edge_states: torch.Tensor,
        edges: torch.LongTensor,
    ):
        concat_states = torch.cat(
            (node_states[edges].view(-1, 2 * self.node_dim), edge_states), axis=1
        )
        if self.residual:
            return self.mask[None, :] * edge_states + self.edge_nn(concat_states)
        else:
            return self.edge_nn(concat_states)


class EquivEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        time_embedding: Optional[FourierEmbedding] = None,
        edge_embedding: Optional[EdgeEmbedding] = None,
        num_layers: int = 4,
        h_input_dim: int = 100,
        smooth_h: bool = True,
        self_conditioning: bool = False, #self-conditioning
        latent_recursion: bool = False, #latent recursion
        M: int = 8, #latent recursion: rows latent dimension
        z_dim: int = 64 #latent dimension: columns latent dimension
    ):
        super(EquivEncoder, self).__init__()

        # Embedding layers
        self.hidden_dim = hidden_dim
        if smooth_h:
            self.node_embedding = nn.Linear(h_input_dim, hidden_dim, bias=False)
        else:
            # we just need to embed the given discrete h
            self.node_embedding = nn.Embedding(h_input_dim + 1, hidden_dim)

        if time_embedding is None:
            time_embedding = FourierEmbedding(1, hidden_dim, trainable=True)

        self.time_embedding = time_embedding
        self.node_time_projection = nn.Linear(
            hidden_dim + time_embedding.out_features, hidden_dim
        )

        if edge_embedding is None:
            edge_embedding = EdgeEmbedding(num_rbf_features=hidden_dim // 2)

        self.edge_embedding = edge_embedding

        # Interaction layers
        self.interactions = nn.ModuleList(
            [
                InteractionLayer(hidden_dim, edge_embedding.out_features)
                for _ in range(num_layers)
            ]
        )

        # Update layers
        self.updates = nn.ModuleList(
            [UpdateLayer(hidden_dim) for _ in range(num_layers)]
        )

        #self-conditioning #TODO: add h-dim for h-conditioning
        self.self_conditioning = self_conditioning
        if self_conditioning:
            self.sc_layer = SelfConditioningResidualLayer(
                node_dim = hidden_dim, 
                edge_dim = edge_embedding.out_features
            )
        
        #latent recursion
        self.latent_recursion = latent_recursion
        if latent_recursion:
            # 1: initialization
            self.z_base = nn.Parameter(torch.randn(M, z_dim)* 0.02) #initialize z_base with small random values
             
            # 2: latent recursion / sync layers. 
            self.latent_sync_modules = nn.ModuleList([
                LatentSyncModule(hidden_dim, z_dim) for _ in range(num_layers + 1) #+1 because we need a module as the primer
            ]) 

            # 3: readout residual MLP
            self.z_residual = nn.Linear(z_dim, z_dim)
            nn.init.zeros_(self.z_residual.weight); nn.init.zeros_(self.z_residual.bias) #0 initialization

    def forward(
        self,
        t: torch.Tensor,
        h: torch.Tensor,
        pos: torch.Tensor,
        node_index: torch.Tensor,
        edge_node_index: Optional[torch.Tensor],
        prev_preds: dict[torch.Tensor, torch.Tensor] = None, #self-conditioning previous predictions
        z_prev: torch.Tensor = None, #latent recursion previous z
    ) -> dict[str, torch.Tensor]:

        t = self.time_embedding(t)
        t_per_atom = t[node_index]

        node_states_v = pos.new_zeros((*pos.shape, self.hidden_dim))
        node_states_s = self.node_embedding(h)
        node_states_s = torch.cat([node_states_s, t_per_atom], dim=1)
        node_states_s = self.node_time_projection(node_states_s)

        edge_states, unit_vectors = self.edge_embedding.forward(
            positions=pos, edge_index=edge_node_index
        )

        #self-conditioning: update node and edge states with previous predictions. Done BEFORE interaction and update layers.
        if self.self_conditioning and prev_preds is not None:
            node_states_s, edge_states = self.sc_layer.forward(
                node_states_s = node_states_s,
                edge_states = edge_states, 
                pos = pos, 
                prev_preds = prev_preds, 
                node_index = node_index, 
                edge_node_index = edge_node_index
            )

        #latent recursion: initialize z_prev and run primer
        Z_proc = z_prev
        z_original = z_prev
        if self.latent_recursion:
            if z_prev is None:
                num_graphs = node_index.max() + 1
                Z_proc = self.z_base.unsqueeze(0).expand(num_graphs, -1, -1)
                z_original = Z_proc
            
            #primer (using module 1)
            Z_proc, node_states_s, node_states_v = self.latent_sync_modules[0](
                z_old = Z_proc,
                s = node_states_s,
                v = node_states_v,
                node_index = node_index
                )
        
        sync_modules = self.latent_sync_modules[1:] if self.latent_recursion else [None] * len(self.interactions) # latent recursion:we need this, otherwise running without latent recursion will throw an error

        for (
            interaction,
            update,
            sync_module
        ) in zip(self.interactions, self.updates, sync_modules):
            node_states_s, node_states_v = interaction.forward(
                node_states_s=node_states_s,
                node_states_v=node_states_v,
                edge_states=edge_states,
                unit_vectors=unit_vectors,
                node_index=node_index,
                edge_node_index=edge_node_index,
            )
            node_states_s, node_states_v = update(node_states_s, node_states_v)
            
            if self.latent_recursion:
                Z_proc, node_states_s, node_states_v = sync_module(
                    z_old = Z_proc,
                    s = node_states_s,
                    v = node_states_v,
                    node_index = node_index
                )
        
        states = {"s": node_states_s, "v": node_states_v}
        
        #latent recursion: final readout with residual connection
        if self.latent_recursion:
            Z_next = z_original + self.z_residual(Z_proc)
            states["z"] = Z_next

        return states
