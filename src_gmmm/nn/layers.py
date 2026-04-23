import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch_scatter import scatter_mean
from torch_geometric.utils import to_dense_batch


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
        features = torch.exp((-((distances - self.offsets) ** 2)) / self.delta)

        if self.cutoff:
            features = features * cosine_cutoff(distances, cutoff=self.max_distance)

        return features

    @property
    def out_features(self):
        return self.num_rbf_features + 1


class EquivLayerNorm(nn.Module):
    def __init__(
        self,
        dims: tuple[int, Optional[int]],
        eps: float = 1e-6,
        affine: bool = True,
    ):
        super().__init__()

        self.dims = dims
        self.sdim, self.vdim = dims
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight_s = nn.Parameter(torch.Tensor(self.sdim))
            self.bias_s = nn.Parameter(torch.Tensor(self.sdim))
            # self.weight_v = nn.Parameter(torch.Tensor(self.vdim))
        else:
            self.register_parameter("weight_s", None)
            self.register_parameter("bias_s", None)
            # self.register_parameter("weight_v", None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            self.weight_s.data.fill_(1.0)
            self.bias_s.data.fill_(0.0)
            # self.weight_v.data.fill_(1.0)

    def forward(
        self, s: torch.Tensor, v: torch.Tensor, index: torch.Tensor
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:

        batch_size = int(index.max()) + 1
        smean = s.mean(dim=-1, keepdim=True)
        smean = scatter_mean(smean, index, dim=0, dim_size=batch_size)

        s = s - smean[index]

        var = (s * s).mean(dim=-1, keepdim=True)
        var = scatter_mean(var, index, dim=0, dim_size=batch_size)
        var = torch.clamp(var, min=self.eps)
        sout = s / var[index]

        if self.affine and self.weight_s is not None and self.bias_s is not None:
            sout = sout * self.weight_s + self.bias_s

        if v is not None:
            vmean = torch.pow(v, 2).sum(dim=1, keepdim=True).mean(dim=-1, keepdim=True)
            vmean = scatter_mean(vmean, index, dim=0, dim_size=batch_size)
            vmean = torch.clamp(vmean, min=self.eps)
            vout = v / vmean[index]

        else:
            vout = None

        out = sout, vout

        return out


#self-conditioning: residual layer that takes previous predictions as input
class SelfConditioningResidualLayer(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int):
        super().__init__()

        self.node_mlp = nn.Sequential(
            nn.Linear(node_dim + 1, node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, node_dim) 
            )

        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim + 1, edge_dim),
            nn.SiLU(),
            nn.Linear(edge_dim, edge_dim)
        )

        

    def forward(self, node_states_s, edge_states, pos , prev_preds, node_index, edge_node_index):
        prev_pos = prev_preds["pos"]
        #prev_h = prev_preds["h"] #TODO: add this back when we start using diffusion_h

        #Nodoes s
        # difference of positions of the same atoms in X_t and X_1 #strategy used by both FlowMol and Harmonic
        node_dist = torch.norm(pos - prev_pos, dim=-1, keepdim=True) 

        #concat current node state, hte predicted h node states and the predicted distance
        node_resid_input = torch.cat([node_states_s, node_dist], dim=-1) #TODO: add prev_h when we start using diffusion_h
        node_states_s = node_states_s + self.node_mlp(node_resid_input) #map back to dim of s 

        #Edges e
        src, dst = edge_node_index
        
        #eucledian length of edges in X_t
        curr_edge_dist = torch.norm(pos[src] - pos[dst], dim=-1, keepdim=True)
         
        #eucledian length of edges in X_1 
        prev_edge_dist = torch.norm(prev_pos[src] - prev_pos[dst], dim=-1, keepdim=True)
        
        #difference in edge lengths of X_1 and X_t
        edge_dist_diff = prev_edge_dist - curr_edge_dist 

        #MLP(concat(edge_states, edge_dist_diff)) + edge states
        edge_resid_input = torch.cat([edge_states, edge_dist_diff], dim=-1)
        edge_states = edge_states + self.edge_mlp(edge_resid_input)

        return node_states_s, edge_states




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


# latent recursion module
class LatentSyncModule(nn.Module):
    def __init__(self, node_dim: int, z_dim:int, num_heads: int = 4, num_blocks: int = 2, skip_transformer_block: bool = False):
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