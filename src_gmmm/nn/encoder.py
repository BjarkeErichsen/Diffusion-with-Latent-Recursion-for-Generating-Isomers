from typing import Optional, Union

import torch
import torch.nn as nn
from torch_scatter import scatter_sum

from ..nn.layers import EdgeEmbedding, EquivLayerNorm, FourierEmbedding
from ..nn.layers import SelfConditioningResidualLayer # self-conditioning
from ..nn.layers import LatentRecursionLayer # latent-recursion
from ..nn.layers import EdgeUpdateLayer # update edge states
from ..nn.layers import RMSNorm, VectorRMSNorm, RepresentationProbe
from ..utils.trm_utilities import SinusoidalPositionalEncoding # latent-recursion


class InteractionLayer(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        condition_dim: Optional[int] = None,
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

        self.ln = EquivLayerNorm(dims=(node_dim, node_dim), condition_dim=condition_dim)

    def forward(
        self,
        node_states_s: torch.Tensor,
        node_states_v: torch.Tensor,
        edge_states: torch.Tensor,
        unit_vectors: torch.Tensor,
        node_index: torch.Tensor,
        edge_node_index: torch.Tensor,
        c: Optional[torch.Tensor] = None,
        skip_residual=False,
    ):
        src_idx, dst_idx = edge_node_index

        node_states_s, node_states_v = self.ln.forward(
            node_states_s, node_states_v, node_index, c=c
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

        if skip_residual:
            return reduced_messages_s, reduced_messages_v
        else:
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

    def forward(self, node_states_s: torch.Tensor, node_states_v: torch.Tensor, skip_residual=False):
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
        if skip_residual:
            return delta_s, delta_v
        else:
            return node_states_s + delta_s, node_states_v + delta_v


class EquivEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        time_embedding: Optional[FourierEmbedding] = None,
        edge_embedding: Optional[EdgeEmbedding] = None,
        num_layers: int = 4,
        h_input_dim: int = 100,
        smooth_h: bool = True,
        self_conditioning: bool = False,  # self-conditioning
        latent_recursion: Union[bool, str, None] = False,  # latent recursion
        M: int = 8,  # latent recursion: rows latent dimension
        z_dim: int = 64,  # latent dimension: columns latent dimension
        update_edge_states: bool = False,  # update edge states
        ablations: dict = None,  # ablations dict
        condition_dim: Optional[int] = None,  # shape condition dimension
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
                InteractionLayer(hidden_dim, edge_embedding.out_features, condition_dim=condition_dim)
                for _ in range(num_layers)
            ]
        )

        # Update layers
        self.updates = nn.ModuleList(
            [UpdateLayer(hidden_dim) for _ in range(num_layers)]
        )

        # self-conditioning
        self.self_conditioning = self_conditioning
        self.ablations = ablations if ablations is not None else {}
        if self_conditioning:
            self.sc_layer = SelfConditioningResidualLayer(
                node_dim=hidden_dim,
                edge_dim=edge_embedding.out_features,
                time_dim=time_embedding.out_features,
                max_distance=edge_embedding.max_distance,
                ablations=self.ablations,
            )

        # latent recursion
        self.latent_recursion = latent_recursion
        is_active_lr = (
            latent_recursion is not None
            and latent_recursion is not False
            and str(latent_recursion).lower() not in ("none", "null", "false", "")
        )
        if is_active_lr:
            self.latent_recursion_layer = LatentRecursionLayer(
                method=latent_recursion,
                hidden_dim=hidden_dim,
                edge_dim=edge_embedding.out_features,
                condition_dim=condition_dim,
                ablations=self.ablations,
            )
        else:
            self.latent_recursion_layer = None

        # update edge states
        self.update_edge_states = update_edge_states
        if update_edge_states:
            self.edge_update_layers = nn.ModuleList(
                [
                    EdgeUpdateLayer(self.hidden_dim, self.edge_embedding.out_features, architecture="layernorm")
                    for _ in range(num_layers - 1)
                ]
            )

        self.probes = nn.ModuleDict({
            "pre_sc_s": RepresentationProbe(),
            "pre_sc_e": RepresentationProbe(),
            "pre_sc_unit_vectors": RepresentationProbe(),
            "post_sc_s": RepresentationProbe(),
            "post_sc_v": RepresentationProbe(),
            "post_sc_e": RepresentationProbe(),
            "post_sc_unit_vectors": RepresentationProbe(),
            "prev_pos": RepresentationProbe(),
            "pos": RepresentationProbe(),
            **{f"layer_{i}_s": RepresentationProbe() for i in range(num_layers)},
            **{f"layer_{i}_v": RepresentationProbe() for i in range(num_layers)},
        })


    def forward(
        self,
        t: torch.Tensor,
        h: torch.Tensor,
        pos: torch.Tensor,
        node_index: torch.Tensor,
        edge_node_index: Optional[torch.Tensor],
        prev_preds: Optional[dict[torch.Tensor, torch.Tensor]] = None,  # self-conditioning previous predictions
        z_prev: Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,  # latent recursion previous z
        c: Optional[torch.Tensor] = None,
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

        if self.latent_recursion_layer is not None:
            node_states_s, node_states_v, edge_states = self.latent_recursion_layer.forward(
                node_states_s=node_states_s,
                node_states_v=node_states_v,
                edge_states=edge_states,
                unit_vectors=unit_vectors,
                node_index=node_index,
                edge_node_index=edge_node_index,
                z_prev=z_prev,
                c=c,
            )

        # self-conditioning: update node and edge states with previous predictions.
        if self.self_conditioning and prev_preds is not None:
            node_states_s = self.probes["pre_sc_s"](node_states_s, node_index)
            edge_states = self.probes["pre_sc_e"](edge_states, edge_node_index)
            unit_vectors = self.probes["pre_sc_unit_vectors"](unit_vectors, edge_node_index)
            prev_preds["pos"] = self.probes["prev_pos"](prev_preds["pos"], node_index)
            pos = self.probes["pos"](pos, node_index)

            node_states_s, node_states_v, edge_states, unit_vectors = self.sc_layer.forward(
                node_states_s=node_states_s,
                node_states_v=node_states_v,
                edge_states=edge_states,
                unit_vectors=unit_vectors,
                pos=pos,
                prev_preds=prev_preds,
                node_index=node_index,
                edge_node_index=edge_node_index,
                t=t_per_atom,
            )

            node_states_s = self.probes["post_sc_s"](node_states_s, node_index)
            node_states_v = self.probes["post_sc_v"](node_states_v, node_index)
            edge_states = self.probes["post_sc_e"](edge_states, edge_node_index)
            unit_vectors = self.probes["post_sc_unit_vectors"](unit_vectors, edge_node_index)

        for i, (
            interaction,
            update,
        ) in enumerate(zip(self.interactions, self.updates)):
            node_states_s, node_states_v = interaction.forward(
                node_states_s=node_states_s,
                node_states_v=node_states_v,
                edge_states=edge_states,
                unit_vectors=unit_vectors,
                node_index=node_index,
                edge_node_index=edge_node_index,
                c=c,
            )
            node_states_s, node_states_v = update(node_states_s, node_states_v)

            if self.update_edge_states and i < len(self.interactions) - 1:
                edge_states = self.edge_update_layers[i](edge_states, node_states_s, node_states_v, edge_node_index)

            node_states_s = self.probes[f"layer_{i}_s"](node_states_s, node_index)
            node_states_v = self.probes[f"layer_{i}_v"](node_states_v, node_index)

        states = {"s": node_states_s, "v": node_states_v, "edge": edge_states}

        # latent recursion: return hidden states for recursion if layer is enabled
        if self.latent_recursion_layer is not None:
            states["z"] = [node_states_s, node_states_v, edge_states]

        return states
