import math
from typing import Optional, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch_scatter import scatter_mean
from torch_geometric.utils import to_dense_batch
from pytorch3d.ops import corresponding_points_alignment

def cosine_cutoff(edge_distances: torch.Tensor, cutoff: float):
    return torch.where(
        edge_distances < cutoff,
        0.5 * (torch.cos(torch.pi * edge_distances / cutoff) + 1.0),
        torch.tensor(0.0, device=edge_distances.device, dtype=edge_distances.dtype),
    )


class FourierEmbedding(nn.Module):
    """
    Random Fourier features (sine and cosine expansion).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        std: float = 1.0,
        trainable: bool = False,
    ):
        super(FourierEmbedding, self).__init__()
        assert (out_features % 2) == 0
        weight = torch.normal(mean=torch.zeros(out_features // 2, in_features), std=std)

        self.trainable = trainable
        if trainable:
            self.weight = nn.Parameter(weight)
        else:
            self.register_buffer("weight", weight)

        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        x = F.linear(x, self.weight)
        cos_features = torch.cos(2 * math.pi * x)
        sin_features = torch.sin(2 * math.pi * x)
        x = torch.cat((cos_features, sin_features), dim=1)

        return x


class EdgeEmbedding(nn.Module):
    def __init__(
        self,
        num_rbf_features: int = 64,
        max_distance: float = 25.0,
        trainable: bool = False,
        norm: bool = True,
        cutoff: bool = True,
    ):
        super().__init__()

        self.norm = norm
        self.num_rbf_features = num_rbf_features
        self.max_distance = max_distance
        self.cutoff = cutoff

        self.register_buffer("delta", torch.tensor(max_distance / num_rbf_features))
        offsets = torch.linspace(
            start=0.0, end=max_distance, steps=num_rbf_features
        ).unsqueeze(0)
        if trainable:
            self.offsets = nn.Parameter(offsets)
        else:
            self.register_buffer("offsets", offsets)

    def forward(
        self,
        positions: torch.Tensor,
        edge_index: torch.Tensor,
        norm: Optional[bool] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        norm = self.norm if norm is None else norm
        dest, source = edge_index

        vectors = positions[dest] - positions[source]  # (n_edges, 3) vector (i - > j)

        distances = torch.sqrt(
            torch.sum(vectors**2, dim=-1, keepdim=True) + 1e-6
        )  # (n_edges, 1)
        d = self.featurize_distances(distances)

        cos = F.cosine_similarity(positions[dest], positions[source], dim=-1).unsqueeze(
            1
        )
        edge_features = torch.cat([d, cos], dim=1)

        if norm:
            vectors = vectors / (distances + 1.0)

        return edge_features, vectors

    def featurize_distances(self, distances: torch.Tensor):
        distances = torch.clamp(distances, 0.0, self.max_distance)
        features = torch.exp((-((distances - self.offsets) ** 2)) / self.delta) #RBF: offsets is 64 values of mu

        if self.cutoff:
            features = features * cosine_cutoff(distances, cutoff=self.max_distance)

        return features

    @property
    def out_features(self):
        return self.num_rbf_features + 1


class EdgeUpdateLayer(nn.Module):
    """
    If using edge updates:
        let edges reflect the new distances By updating the edges during each message passing layer
    """
    def __init__(self, hidden_dim: int, edge_dim: int, architecture: str="default"):
        super().__init__()

        self.architecture = architecture

        if self.architecture == "default":
            self.v_to_edge_mlp = nn.Linear(2 * hidden_dim, edge_dim, bias=False)
            self.edge_mlp = nn.Sequential(
                nn.Linear(2 * edge_dim, edge_dim),
                nn.SiLU(),
                nn.Linear(edge_dim, edge_dim)
            )
            
        
    def forward(self, edge_states, node_states_s, node_states_v, edge_index):
        dest, source = edge_index

        dest_v = node_states_v[dest]   # (n_edges, 3, z_dim)
        source_v = node_states_v[source] # (n_edges, 3, z_dim)
        source_s = node_states_s[source] # (n_edges, z_dim)
        dest_s = node_states_s[dest] # (n_edges, z_dim)

        if self.architecture == "default":
            distances = torch.norm(dest_v - source_v, dim=1)  # (E, hidden_dim)
            cos_sim = nn.functional.cosine_similarity(dest_v, source_v, dim=1)  # (E, hidden_dim)
            v_geom_features = torch.cat([distances, cos_sim], dim=-1)  # (E, 2 * hidden_dim)
            edge_from_v = self.v_to_edge_mlp(v_geom_features)  # (E, edge_dim)
            edge_residual = edge_states - edge_from_v  # (E, edge_dim)
            edge_states = edge_states + self.edge_mlp(torch.cat([edge_states, edge_residual], dim=-1))  # (E, edge_dim)
        return edge_states


class EquivLayerNorm(nn.Module):
    def __init__(
        self,
        dims: tuple[int, Optional[int]],
        eps: float = 1e-6,
        affine: bool = True,
        condition_dim: Optional[int] = None, 
    ):
        super().__init__()

        self.dims = dims
        self.sdim, self.vdim = dims
        self.eps = eps
        self.affine = affine
        self.condition_dim = condition_dim

        if affine:
            self.weight_s = nn.Parameter(torch.Tensor(self.sdim))
            self.bias_s = nn.Parameter(torch.Tensor(self.sdim))
            # self.weight_v = nn.Parameter(torch.Tensor(self.vdim))
        else:
            self.register_parameter("weight_s", None)
            self.register_parameter("bias_s", None)
            # self.register_parameter("weight_v", None)

        if self.condition_dim is not None:
            self.cond_proj_s = nn.Linear(condition_dim, 2 * self.sdim)
            nn.init.zeros_(self.cond_proj_s.weight)
            nn.init.zeros_(self.cond_proj_s.bias)
            
            if self.vdim is not None:
                self.cond_proj_v = nn.Linear(condition_dim, self.vdim)
                nn.init.zeros_(self.cond_proj_v.weight)
                nn.init.zeros_(self.cond_proj_v.bias)
                
        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            self.weight_s.data.fill_(1.0)
            self.bias_s.data.fill_(0.0)
            # self.weight_v.data.fill_(1.0)

    def forward(
        self, s: torch.Tensor, v: torch.Tensor, index: torch.Tensor, c: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:

        batch_size = int(index.max()) + 1
        smean = s.mean(dim=-1, keepdim=True)
        smean = scatter_mean(smean, index, dim=0, dim_size=batch_size)

        s = s - smean[index]

        var = (s * s).mean(dim=-1, keepdim=True)
        var = scatter_mean(var, index, dim=0, dim_size=batch_size)
        var = torch.clamp(var, min=self.eps)
        sout = s / torch.sqrt(var[index]) #modified from var[index]

        if self.affine and self.weight_s is not None and self.bias_s is not None:
            sout = sout * self.weight_s + self.bias_s
            
        # Apply S conditioning
        if self.condition_dim is not None and c is not None:
            gamma_s, beta_s = torch.chunk(self.cond_proj_s(c), chunks=2, dim=-1)
            sout = sout * (1 + gamma_s[index]) + beta_s[index]
        
        if v is not None:
            vmean = torch.pow(v, 2).sum(dim=1, keepdim=True).mean(dim=-1, keepdim=True)
            vmean = scatter_mean(vmean, index, dim=0, dim_size=batch_size) #vmean -> (n_graphs, 1, 1)
            vmean = torch.clamp(vmean, min=self.eps) 
            vout = v / torch.sqrt(vmean[index]) #modified from vmean[index]
            
            # Apply V conditioning
            if self.condition_dim is not None and c is not None:
                vout = vout * (1 + self.cond_proj_v(c)[index].unsqueeze(1))
        else:
            vout = None

        out = sout, vout

        return out

#self-conditioning: residual layer that takes previous predictions as input
class SelfConditioningResidualLayer(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, time_dim: int, max_distance: float = 8, ablations: dict = None):
        super().__init__()
        self.node_dim = node_dim
        self.max_distance = max_distance
        self.ablations = ablations if ablations is not None else {}
        
        
        self.rbf_s_dim = 64
        self.rbf_edge_dim = 16

        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim + self.rbf_s_dim, node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, node_dim) 
            )

        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim + self.rbf_edge_dim, edge_dim),
            nn.SiLU(),
            nn.Linear(edge_dim, edge_dim)
        )
        
        self.v_mlp = nn.Linear(node_dim, node_dim, bias=False) #only used if we want to use v-conditioning

        self.unit_vector_mlp = nn.Linear(2, 1, bias=False)
        
        #gates 
        self.s_gate = nn.Sequential(
            nn.Linear(time_dim, 1),
            nn.Sigmoid()
        )
        self.v_gate = nn.Sequential(
            nn.Linear(time_dim, 1),
            nn.Sigmoid()
        )
        self.edge_gate = nn.Sequential(
            nn.Linear(time_dim, 1),
            nn.Sigmoid()
        )
        self.unit_vector_gate = nn.Sequential(
            nn.Linear(time_dim, 1),
            nn.Sigmoid()
        )

        self.probes = nn.ModuleDict({
            "sc_s_gate": RepresentationProbe(),
            "sc_edge_gate": RepresentationProbe(),
            "sc_v_gate": RepresentationProbe(),
            "sc_unit_vector_gate": RepresentationProbe(),
        })

        # RBF buffers
        self.register_buffer("offsets_s", torch.linspace(0, max_distance, self.rbf_s_dim).unsqueeze(0))
        self.register_buffer("offsets_edge", torch.linspace(0, max_distance, self.rbf_edge_dim).unsqueeze(0))
        self.register_buffer("delta_s", torch.tensor(max_distance / self.rbf_s_dim))
        self.register_buffer("delta_edge", torch.tensor(max_distance / self.rbf_edge_dim))


    def featurize_distances(self, distances: torch.Tensor, offsets: torch.Tensor, delta: torch.Tensor):
        distances = torch.clamp(distances, 0, self.max_distance)
        return torch.exp(-((distances - offsets)**2) / delta)


    def forward(self, node_states_s, node_states_v, edge_states, unit_vectors, pos, prev_preds, node_index, edge_node_index, t):
        prev_pos = prev_preds["pos"]
        #prev_h = prev_preds["h"] #TODO: add this back when we start using diffusion_h

        if False: #alignment
            pos_dense, mask = to_dense_batch(pos, node_index)
            prev_pos_dense, _ = to_dense_batch(prev_pos, node_index)
            alignment = corresponding_points_alignment(
                prev_pos_dense, 
                pos_dense, 
                weights=mask.float(), 
                estimate_scale=False
            )
            prev_pos_dense_aligned = torch.bmm(prev_pos_dense, alignment.R) #+ alignment.T.unsqueeze(1) #no rotation.
            prev_pos = prev_pos_dense_aligned[mask]

        #tracking alignment properties
        #R = alignment.R
        #trace = torch.diagonal(input=R, dim1=-2, dim2=-1).sum(-1)
        #cos_theta = ((trace - 1) / 2).clamp(-1.0, 1.0)
        #theta = torch.acos(cos_theta)
        #print(torch.abs(theta).mean())
        #print(torch.abs(torch.norm(alignment.T.unsqueeze(1), dim=-1, keepdim=True)).mean())

        if not self.ablations.get("ablate_s", False):
            #Nodoes s
            # difference of positions of the same atoms in X_t and X_1 #strategy used by both FlowMol and Harmonic
            node_dist = torch.norm(pos - prev_pos, dim=-1, keepdim=True) 
            node_dist_rbf = self.featurize_distances(node_dist, self.offsets_s, self.delta_s)
            node_resid_input = torch.cat([node_states_s, node_dist_rbf], dim=-1) #TODO: add prev_h when we start using diffusion_h
            #gate_val = self.s_gate(t)
            #gate_val = self.probes["sc_s_gate"](gate_val, node_index)
            gate_val = 1
            node_states_s = node_states_s + gate_val * self.node_mlp(node_resid_input) #map back to dim of s 

        if not self.ablations.get("ablate_edge", False):
            #Edges e
            src, dst = edge_node_index
            
            #eucledian length of edges in X_t
            curr_edge_dist = torch.norm(pos[src] - pos[dst], dim=-1, keepdim=True)
            curr_edge_dist  = self.featurize_distances(curr_edge_dist, self.offsets_edge, self.delta_edge) # FIX: Changed `sself.offsets_edge` to `self.delta_edge` (typo + wrong variable for the delta parameter)

            #eucledian length of edges in X_1 
            prev_edge_dist = torch.norm(prev_pos[src] - prev_pos[dst], dim=-1, keepdim=True)
            prev_edge_dist = self.featurize_distances(prev_edge_dist, self.offsets_edge, self.delta_edge)

            #difference in edge lengths of X_1 and X_t
            edge_dist_diff = prev_edge_dist - curr_edge_dist 
            #edge_dist_rbf = self.featurize_distances(torch.abs(edge_dist_diff), self.offsets_edge, self.delta_edge)

            #MLP(concat(edge_states, edge_dist_diff)) + edge states
            edge_resid_input = torch.cat([edge_states, edge_dist_diff], dim=-1)
            #gate_val = self.edge_gate(t[src])
            #gate_val = self.probes["sc_edge_gate"](gate_val, edge_node_index)
            gate_val = 1
            edge_states = edge_states + gate_val * self.edge_mlp(edge_resid_input)
        
        if not self.ablations.get("ablate_unit_vectors", False):
            """
            Updated to compute unit vectors on x0_prev, similar to how its computed on xt
            """
            
            vectors = prev_pos[src] - prev_pos[dst]  

            distances = torch.sqrt(
                torch.sum(vectors**2, dim=-1, keepdim=True) + 1e-6
            )  
            unit_vector_x0_prev = vectors / (distances + 1.0)

            unit_vector_residual = unit_vector_x0_prev - unit_vectors

            
            unit_vector_mlp_out = torch.stack([unit_vectors, unit_vector_residual], dim=-1)
            unit_vectors = unit_vectors + self.unit_vector_mlp(unit_vector_mlp_out).squeeze(-1)

        if not self.ablations.get("ablate_v", False):
            # Use aligned prev_pos for vector orientation
            v = (pos - prev_pos) / (torch.norm(pos - prev_pos, dim=-1, keepdim=True) + 1e-12) 
            v = v.unsqueeze(-1).expand(-1, -1, self.node_dim) # v (N x 3) -> (N x 3 x F) 
            #gate_val = self.v_gate(t).unsqueeze(1)
            #gate_val = self.probes["sc_v_gate"](gate_val, node_index)
            gate_val = 1 
            node_states_v = node_states_v + gate_val * self.v_mlp(v) # unsqueeze(1) for broadcasting [N, 1, 1] with [N, 3, F]

        return node_states_s, node_states_v, edge_states, unit_vectors


class RepresentationProbe(nn.Module):
    """
    Dummy layer used purely as an anchor for PyTorch forward hooks.
    Takes the tensor and its batch index, returning the tensor unmodified.
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, x, node_index=None):
        return x


#ONLY scaling, no bias
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scaling = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # Normalize across the feature dimension (z_dim)
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.scaling


class VectorRMSNorm(nn.Module):
    """
    SE(3)-Equivariant RMSNorm for Vector Features v of shape (N, 3, hidden_dim).
    Normalizes across spatial (3D) and hidden feature dimensions to maintain rotational invariance.
    """
    def __init__(self, hidden_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scaling = nn.Parameter(torch.ones(hidden_dim))

        
    def forward(self, v: torch.Tensor, s: torch.Tensor = None) -> torch.Tensor:
        # v shape: (N, 3, hidden_dim)
        # s shape: (N, scalar_hidden_dim) - optional scalar features
        
        norm_sq = torch.sum(v**2, dim=-2, keepdim=True) # (N, 1, hidden_dim)
        rms = torch.sqrt(torch.mean(norm_sq, dim=-1, keepdim=True) + self.eps)
        v_normed = v / rms
        
        # gate is a learned function (e.g., Linear + Sigmoid) over the invariant norm_sq
        # gate shape: (N, 1, hidden_dim)
        #gate = self.gate_network(norm_sq) 
        
        return v_normed 



# latent recursion module
class LatentSyncModule(nn.Module):
    def __init__(self, node_dim: int, z_dim:int, num_heads: int = 4, num_blocks: int = 2, skip_transformer_block: bool = True):
        super().__init__()

        self.node_dim, self.z_dim = node_dim, z_dim
        self.skip_transformer_block = skip_transformer_block

        # 1 Read: Used to inject information from s into z (with a residual connection)
        self.read_attn = nn.MultiheadAttention(embed_dim=z_dim, kdim=node_dim, vdim=node_dim, num_heads=num_heads, batch_first=True)
        self.read_mlp = nn.Linear(z_dim, z_dim)

        # 2 Compute
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=z_dim,
            nhead=4,
            dim_feedforward=z_dim * 2,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True 
        )
        self.transformer_blocks = nn.TransformerEncoder(self.encoder_layer, num_layers=num_blocks)
        
        # 3 Write: Used to inject information from z into s (with a residual connection)
        self.write_attn = nn.MultiheadAttention(embed_dim=node_dim, kdim=z_dim, vdim=z_dim, num_heads=4, batch_first=True)
        self.write_mlp = nn.Linear(node_dim, node_dim)

        #layer norm on each token of z

        #RMSNorm layers
        self.norm_read = RMSNorm(z_dim)
        self.norm_compute = RMSNorm(z_dim)

    def forward(self, z_old: torch.Tensor, s: torch.Tensor, v: torch.Tensor, node_index: torch.Tensor):
        
        #to dense representation
        s_dense, mask = to_dense_batch(s, node_index) # [B, Max_N, node_dim]
        
        # 1 Read:  information s -> z
        o_read, _ = self.read_attn(z_old, s_dense, s_dense, key_padding_mask=~mask)
        o_read_norm = self.norm_read(self.read_mlp(o_read)) 
        
        z_new = z_old + o_read_norm #residual connection is kept intact -> no completicated identity path

        # 2 Compute: z -> z
        if self.skip_transformer_block:
            z_proc = z_new
        else:
            z_proc = self.transformer_blocks(z_new)
    
        #Normalize output of transformer #TODO maybe not needed
        z_proc = self.norm_compute(z_proc)

        # 3 Write:  information z -> s
        o_write, _ = self.write_attn(s_dense, z_proc, z_proc)
        
        s_new_dense = s_dense + self.write_mlp(o_write) # [B, N_max, node_dim]  #norms = torch.linalg.vector_norm(self.write_mlp(o_write), ord=2, dim=(1, 2))

        # back to sparse
        s_new = s_new_dense[mask]

        # TODO: Later add vector processing as well. Currently we just pass v through unchanged.
        v_new = v
        
        return z_proc, s_new, v_new


class LatentRecursionLayer(nn.Module):
    """
    Dedicated specialized layer for Latent Recursion conditioning.
    Determined by the method name string passed via `method`.
    """
    def __init__(
        self,
        method: Union[str, bool],
        hidden_dim: int,
        edge_dim: int,
        condition_dim: Optional[int] = None,
        ablations: Optional[dict] = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim
        self.ablations = ablations if ablations is not None else {}

        # Normalize method string
        if isinstance(method, bool):
            self.method = "edge_v_update_s_v" if method else "none"
        else:
            self.method = str(method).lower()

        # Import InteractionLayer & UpdateLayer from encoder
        from .encoder import InteractionLayer, UpdateLayer

        # Instantiate required components based on method string
        if self.method in ("v_identity", "s_baseline", "v_identity_s_baseline", "edge_v_v_identity_s_baseline"):
            self.s_sync = nn.Linear(hidden_dim, hidden_dim, bias=True)

        if self.method in ("update_layer", "full_mp_layer", "edge_v_update_s_v", "true"):
            self.update_lr = UpdateLayer(hidden_dim)

        if self.method in ("full_mp_layer", "mp_edge_from_v"):
            self.interaction = InteractionLayer(hidden_dim, edge_dim, condition_dim=condition_dim)

        if self.method in (
            "edge_from_v",
            "mp_edge_from_v",
            "edge_v_v_identity_s_baseline",
            "edge_v_v_scaled_s_normed",
            "edge_v_update_s_v",
            "true",
        ):
            self.v_to_edge_mlp = nn.Linear(2 * hidden_dim, edge_dim, bias=False)
            self.edge_mlp = nn.Sequential(
                nn.Linear(2 * edge_dim, edge_dim),
                nn.SiLU(),
                nn.Linear(edge_dim, edge_dim),
            )

        if self.method == "edge_v_v_scaled_s_normed":
            self.s_sync = nn.Linear(hidden_dim, hidden_dim, bias=True)
            self.s_norm = RMSNorm(hidden_dim)
            self.v_norm = VectorRMSNorm(hidden_dim)
            self.e_norm = RMSNorm(edge_dim)

        # =========================================================================
        # OLD CODE (Preserved as comments for reference, not instantiated)
        # =========================================================================
        # self.z_base = nn.Parameter(torch.randn(M, z_dim) * 0.02)
        # self.latent_sync_modules = nn.ModuleList([LatentSyncModule(hidden_dim, z_dim) for _ in range(num_layers + 1)])
        # self.z_residual = nn.Linear(z_dim, z_dim)
        # self.z_pe = SinusoidalPositionalEncoding(d_model=z_dim, max_len=M)
        # self.s_sync = nn.Linear(hidden_dim, hidden_dim, bias=True)
        # self.s_gat = nn.Linear(hidden_dim, hidden_dim)
        # self.write_attn = nn.MultiheadAttention(embed_dim=hidden_dim, kdim=hidden_dim, vdim=hidden_dim, num_heads=4, batch_first=True)
        # self.v_sync = nn.Linear(hidden_dim, hidden_dim, bias=False)
        # self.v_gat = nn.Linear(16, hidden_dim)
        # self.v_gat_s = nn.Linear(hidden_dim, hidden_dim)
        # self.edge_sync = nn.Linear(edge_dim, edge_dim, bias=False)
        # self.s_mlp = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim, bias=False))
        # self.e_mlp = nn.Sequential(nn.Linear(edge_dim * 2, edge_dim, bias=False))
        # self.s_norm = RMSNorm(hidden_dim)
        # self.e_norm = RMSNorm(edge_dim)

    def forward(
        self,
        node_states_s: torch.Tensor,
        node_states_v: torch.Tensor,
        edge_states: torch.Tensor,
        unit_vectors: torch.Tensor,
        node_index: torch.Tensor,
        edge_node_index: torch.Tensor,
        z_prev: Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
        c: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if z_prev is None:
            return node_states_s, node_states_v, edge_states

        lr_s = z_prev[0]
        lr_v = z_prev[1]

        if self.method == "v_identity":
            ### latent recursion only v identity mapping
            node_states_v = lr_v

        elif self.method == "s_baseline":
            ### latent recursion only s baseline
            node_states_s = node_states_s + self.s_sync(lr_s)

        elif self.method == "v_identity_s_baseline":
            ### Latent recursion v identity mapping s baseline
            node_states_v = lr_v
            node_states_s = node_states_s + self.s_sync(lr_s)

        elif self.method == "update_layer":
            ### update layer (good mixing logic)
            delta_s, delta_v = self.update_lr(lr_s, lr_v, skip_residual=True)
            node_states_s = node_states_s + delta_s
            node_states_v = node_states_v + delta_v

        elif self.method == "full_mp_layer":
            ### full mp layer (same order as reference model)
            delta_s, delta_v = self.interaction(
                lr_s, lr_v, edge_states, unit_vectors, node_index, edge_node_index, c=c, skip_residual=True
            )
            node_states_s = node_states_s + delta_s
            node_states_v = node_states_v + delta_v
            node_states_s, node_states_v = self.update_lr(node_states_s, node_states_v, skip_residual=False)

        elif self.method == "edge_from_v":
            ### edge from v
            dest, source = edge_node_index
            v_dest = lr_v[dest]
            v_src = lr_v[source]
            distances = torch.norm(v_dest - v_src, dim=1)
            cos_sim = F.cosine_similarity(v_dest, v_src, dim=1)
            v_geom_features = torch.cat([distances, cos_sim], dim=-1)
            edge_from_v = self.v_to_edge_mlp(v_geom_features)
            edge_residual = edge_states - edge_from_v
            edge_states = edge_states + self.edge_mlp(torch.cat([edge_states, edge_residual], dim=-1))

        elif self.method == "mp_edge_from_v":
            ### MP + edge from v
            dest, source = edge_node_index
            v_dest = lr_v[dest]
            v_src = lr_v[source]
            distances = torch.norm(v_dest - v_src, dim=1)
            cos_sim = F.cosine_similarity(v_dest, v_src, dim=1)
            v_geom_features = torch.cat([distances, cos_sim], dim=-1)
            edge_from_v = self.v_to_edge_mlp(v_geom_features)
            edge_residual = edge_states - edge_from_v
            edge_states = edge_states + self.edge_mlp(torch.cat([edge_states, edge_residual], dim=-1))

            delta_s, delta_v = self.interaction(
                lr_s, lr_v, edge_states, unit_vectors, node_index, edge_node_index, c=c, skip_residual=True
            )
            node_states_s = node_states_s + delta_s
            node_states_v = node_states_v + delta_v
            node_states_s, node_states_v = self.update_lr(node_states_s, node_states_v, skip_residual=False)

        elif self.method == "edge_v_v_identity_s_baseline":
            ### edge from v ANDv identity mapping s baseline
            node_states_v = lr_v
            node_states_s = node_states_s + self.s_sync(lr_s)
            dest, source = edge_node_index
            v_dest = lr_v[dest]
            v_src = lr_v[source]
            distances = torch.norm(v_dest - v_src, dim=1)
            cos_sim = F.cosine_similarity(v_dest, v_src, dim=1)
            v_geom_features = torch.cat([distances, cos_sim], dim=-1)
            edge_from_v = self.v_to_edge_mlp(v_geom_features)
            edge_residual = edge_states - edge_from_v
            edge_states = edge_states + self.edge_mlp(torch.cat([edge_states, edge_residual], dim=-1))

        elif self.method == "edge_v_v_scaled_s_normed":
            ### edge from v AND v scaled mapping s normed
            node_states_v = self.v_norm(lr_v)
            node_states_s = node_states_s + self.s_norm(self.s_sync(lr_s))
            dest, source = edge_node_index
            v_dest = lr_v[dest]
            v_src = lr_v[source]
            v_diff = v_dest - v_src
            distances = torch.sqrt(torch.sum(v_diff ** 2, dim=1) + 1e-8)
            cos_sim = F.cosine_similarity(v_dest, v_src, dim=1)
            v_geom_features = torch.cat([distances, cos_sim], dim=-1)
            edge_from_v = self.v_to_edge_mlp(v_geom_features)
            edge_residual = edge_states - edge_from_v
            edge_update = self.edge_mlp(torch.cat([edge_states, edge_residual], dim=-1))
            edge_states = edge_states + self.e_norm(edge_update)

        elif self.method in ("edge_v_update_s_v", "true"):
            ### edge from v AND update layer
            delta_s, delta_v = self.update_lr(lr_s, lr_v, skip_residual=True)
            node_states_s = node_states_s + delta_s
            node_states_v = node_states_v + delta_v

            dest, source = edge_node_index
            v_dest = lr_v[dest]
            v_src = lr_v[source]
            distances = torch.norm(v_dest - v_src, dim=1)
            cos_sim = F.cosine_similarity(v_dest, v_src, dim=1)
            v_geom_features = torch.cat([distances, cos_sim], dim=-1)
            edge_from_v = self.v_to_edge_mlp(v_geom_features)
            edge_residual = edge_states - edge_from_v
            edge_states = edge_states + self.edge_mlp(torch.cat([edge_states, edge_residual], dim=-1))

        else:
            raise ValueError(f"Unknown latent recursion method: '{self.method}'")

        # =========================================================================
        # OLD CODE (Preserved as comments for reference, not executed)
        # =========================================================================
        # #baseline s, v
        # #node_states_s = node_states_s + self.s_sync(z_prev[0])
        # #node_states_v = node_states_v + self.v_sync(z_prev[1])

        # #GLU without v
        # #node_states_s = node_states_s + self.s_sync(z_prev[0]) * torch.sigmoid(self.s_gat(node_states_s))

        # #Multihead attention s
        # #node_states_s = node_states_s + self.write_attn(query=node_states_s, key=z_prev, value=z_prev)[0]

        # #SwiGLU s
        # #node_states_s = node_states_s + self.s_sync(z_prev[0]) * torch.sigmoid(self.s_gat(node_states_s)) * self.s_gat(node_states_s)

        # #GLU with s and v
        # #node_states_s = node_states_s + self.s_sync(z_prev[0]) * torch.sigmoid(self.s_gat(node_states_s))
        # #node_states_v = node_states_v + self.v_sync(z_prev[1]) * torch.sigmoid(self.v_gat_s(z_prev[0])).unsqueeze(1)

        # #normalized GLU with s and v
        # #node_states_s = node_states_s + self.s_sync(z_prev[0]) * torch.sigmoid(self.s_gat(node_states_s))
        # #node_states_v = node_states_v + self.v_sync(z_prev[1]) * torch.sigmoid(self.v_gat_s(z_prev[0])).unsqueeze(1)

        # #baseline s, v edge_embed
        # #node_states_s = node_states_s + self.s_sync(z_prev[0])
        # #node_states_v = node_states_v + self.v_sync(z_prev[1])
        # #edge_states = edge_states + self.edge_sync(z_prev[2])

        # #residual for s and edge_states
        # #s_residual = node_states_s - z_prev[0]
        # #s_concat = torch.cat([node_states_s, s_residual], dim=-1)
        # #node_states_s = node_states_s + self.s_norm(self.s_mlp(s_concat))
        # #e_residual = edge_states - z_prev[2]
        # #e_concat = torch.cat([edge_states, e_residual], dim=-1)
        # #edge_states = edge_states + self.e_norm(self.e_mlp(e_concat))
        # #node_states_v = node_states_v + self.v_sync(z_prev[1])

        return node_states_s, node_states_v, edge_states