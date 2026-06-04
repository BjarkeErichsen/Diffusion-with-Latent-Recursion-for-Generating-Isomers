# TRM_g3m Self-Conditioning Implementation Plan

This document details the exact changes needed to incorporate FlowMol-like self-conditioning into `TRM_g3m`. 

## Concept: How FlowMol Implements Self-Conditioning

FlowMol implements self-conditioning by routing the predicted endpoints ($x_{1\_pred}$, $h_{pred}$) back into the intermediate layers of the equivariant network. 
The implementation defines specifically **where** in the architecture the conditioning is injected, **how** the features are constructed to preserve equivariance, and **when** it is activated during training.

### 1. Architecture Injection Strategy

The self-conditioning injection does *not* happen iteratively within every message-passing step. Instead, it is executed exactly once per forward pass: **immediately after the initial node and edge embeddings are calculated, but before the main integration convolutions (the repeated interaction/update layers).** 

This provides a single, enriched starting point representing the global topological and geometric "blueprint" before the expensive message-passing layers refine the local interactions.

*Pseudo-code of the Encoder Forward Pass:*
```python
def encoder_forward(pos_t, h_t, prev_preds):
    # 1. Base embedding (STANDARD MODEL BEHAVIOR)
    node_embeds = embed_nodes(h_t)
    edge_embeds = embed_edges(pos_t)
    
    # 2. Self-Conditioning Injection (NEW BEHAVIOR)
    if prev_preds is not None:
        # Inject the context of destination directly into the initial embeddings
        node_embeds, edge_embeds = residual_layer(
            node_embeds, edge_embeds, pos_t, prev_preds
        )
    
    # 3. Message Passing GNN Loop (STANDARD MODEL BEHAVIOR)
    for interaction_layer, update_layer in layers:
        node_embeds, edge_embeds = interaction_layer(node_embeds, edge_embeds, pos_t)
        node_embeds = update_layer(node_embeds)
        
    return predict_endpoints(node_embeds, edge_embeds)
```

### 2. Feature Construction in the Residual Layer (Data Tracks)

To merge a 3D coordinate prediction ($x_{1\_pred}$) with an SE(3)-equivariant Graph Neural Network without breaking rotational equivariance, FlowMol introduces a specialized **Residual Layer** that computes purely invariant scalars ($\Delta d$) derived from `prev_preds`.

The self-conditioning operates on a **two-track** structure: an invariant scalar node track ($s$) and an invariant scalar edge track ($e$). It explicitly extracts data spanning **both** the continuous geometric space (positions) and the discrete semantic semantic space (atom types and charges, $h$).

**A. Node Track ($s$) Injection (Positions & Atom Types):** 
Instead of concatenating raw coordinates into the $s$ track, we compute the equivariant Euclidean distance between a node's current varying position $x_t$ and its constant predicted destination $x_{1\_pred}$. 
$$ d_{node} = \|x_t - x_{1\_pred}\|_2 $$
This geometric scalar distance ($d_{node}$), along with the explicitly predicted categorical semantic state ($h_{pred}$ representing atom types and charges), is non-linearly combined via an MLP into the base scalar node embeddings (`node_embeds`). Thus, the node track consumes both positional shifts and targeted atom categories.

**B. Edge Track ($e$) Injection (Relative Geometry):**
For the $e$ track, we measure the scalar difference between the bond length in the current noisy state ($d_{t, edge}$) and the bond length in the predicted final state ($d_{1\_pred, edge}$).
$$ \Delta d_{edge} = d_{1\_pred, edge} - d_{t, edge} $$
This relative scalar difference is injected into the scalar edge embeddings (`edge_embeds`) via an MLP, explicitly signaling whether a bond must stretch or compress.

*FlowMol-like Residual Injection snippet:*
```python
# A. Node Invariance (distance to target destination)
node_dist_to_target = torch.norm(pos_t - prev_preds["pos"], dim=-1, keepdim=True)
node_resid_input = torch.cat([node_embeds, prev_preds["h"], node_dist_to_target], dim=-1)
node_embeds = node_embeds + node_mlp(node_resid_input)

# B. Edge Invariance (change in distance between connected atoms)
curr_edge_dist = torch.norm(pos_t[src] - pos_t[dst], dim=-1, keepdim=True)
prev_edge_dist = torch.norm(prev_preds["pos"][src] - prev_preds["pos"][dst], dim=-1, keepdim=True)
edge_dist_change = prev_edge_dist - curr_edge_dist

edge_resid_input = torch.cat([edge_embeds, edge_dist_change], dim=-1)
edge_embeds = edge_embeds + edge_mlp(edge_resid_input)
```

### 3. Training Strategy (The Dummy Pass)

To prepare the network to receive `prev_preds` while avoiding over-reliance or duplicating compute symmetrically, FlowMol evaluates a stochastic **dummy pass**.

For every training iteration, a coin flip (e.g., $p=0.5$) determines if self-conditioning is heavily punished or trained:
- **Conditioning drops out:** `prev_preds = None`. The model is forced to predict endpoints blindly from $x_t$.
- **Conditioning activates:** A `torch.no_grad()` dummy pass simulates the flawed initial prediction. This prediction is fed back into the true, gradient-tracked forward pass.

*FlowMol-like Training snippet:*
```python
def training_step(batch, t, scprop=0.5):
    prev_preds = None
    
    # Generate the dummy prior guess without tracking gradients
    if self_conditioning and torch.rand(1).item() > scprop:
        with torch.no_grad():
            prev_preds = model(t, pos_t, h_t)
    
    # Calculate loss on the actual gradient-tracked pass
    final_preds = model(t, pos_t, h_t, prev_preds=prev_preds)
    return compute_loss(final_preds, ground_truth)
```


Below are the **exact line-by-line file changes** required. I have explicitly labeled any code snippet that modifies existing code with **[EXISTING CODE]** at the start of original lines, and **[NEW CODE]** for lines you must add. 

---

### Phase 1: Creating the Residual Layer

**Target File:** `src_gmmm/nn/layers.py`  
**Where:** Append to the very bottom of the file (Line 129).  
**Action:** Add the following complete `SelfConditioningResidualLayer` class. This is entirely **[NEW CODE]**. 
*Note:* This code is fully syntactically valid and compiles natively using PyTorch layers.

```python
# [NEW CODE] - Add to the bottom of src_gmmm/nn/layers.py
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

        #TODO: Maybe add self-conditioning for node states v as well?

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
```

---

### Phase 2: Updating the Encoder

**Target File:** `src_gmmm/nn/encoder.py`  
**Where:** Update `EquivEncoder` `__init__` and `forward` logic.  
**Action:** Import the residual layer, instantiate it in `__init__`, and use it inside `forward`. 

Lines 6-8:
```python
# [EXISTING CODE]
from ..nn.layers import EdgeEmbedding, EquivLayerNorm, FourierEmbedding
# [NEW CODE]
from ..nn.layers import SelfConditioningResidualLayer #self-conditioning
```

Lines ~137-140 (`EquivEncoder.__init__` arguments):
```python
# [EXISTING CODE]
        num_layers: int = 4,
        h_input_dim: int = 100,
        smooth_h: bool = True,
# [NEW CODE]
        self_conditioning: bool = False, #self-conditioning
```

Lines ~173 (`EquivEncoder.__init__` body, append this at the end of the init):
```python
# [EXISTING CODE]
        self.updates = nn.ModuleList(
            [UpdateLayer(hidden_dim) for _ in range(num_layers)]
        )
# [NEW CODE]
        #self-conditioning #TODO: add h-dim for h-conditioning
        self.self_conditioning = self_conditioning
        if self_conditioning:
            self.sc_layer = SelfConditioningResidualLayer(
                node_dim = hidden_dim, 
                edge_dim = edge_embedding.out_features
            )
```

 Lines ~176-184 (`EquivEncoder.forward` signature):
```python
# [EXISTING CODE]
    def forward(
        self,
        t: torch.Tensor,
        h: torch.Tensor,
        pos: torch.Tensor,
        node_index: torch.Tensor,
        edge_node_index: Optional[torch.Tensor],
# [NEW CODE]
        prev_preds: dict[torch.Tensor, torch.Tensor] = None, #self-conditioning previous predictions
# [EXISTING CODE]
    ) -> dict[str, torch.Tensor]:
```

Lines ~194-196 (`EquivEncoder.forward` body, right below `edge_embedding.forward` and before the `for` loop of interactions):
```python
# [EXISTING CODE]
        edge_states, unit_vectors = self.edge_embedding.forward(
            positions=pos, edge_index=edge_node_index
        )
# [NEW CODE]
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

# [EXISTING CODE]
        for (
            interaction,
            update,
        ) in zip(self.interactions, self.updates):
```

---

### Phase 3: Updating the Score Wrapper

**Target File:** `src_gmmm/model/score.py`  
**Where:** Inside `EquivariantParameterization.forward`.  
**Action:** Pass `prev_preds` smoothly down from the diffusion module into the encoder.

Lines ~18-25 (`EquivariantParameterization.forward`):
```python
# [EXISTING CODE]
    def forward(
        self,
        t: torch.Tensor,
        h: torch.Tensor,
        pos: torch.Tensor,
        node_index: torch.Tensor,
        edge_node_index: torch.Tensor,
# [NEW CODE]
        prev_preds = None #self-conditioning: previous predictions
# [EXISTING CODE]
    ):
        states = self.encoder.forward(
            t=t,
            h=h,
            pos=pos,
            node_index=node_index,
            edge_node_index=edge_node_index,
# [NEW CODE]
            prev_preds=prev_preds, #self-conditioning: pass previous predictions to the model
# [EXISTING CODE]
        )
```

---

### Phase 4: Triggering Self-Conditioning in Diffusion Loop

**Target File:** `src_gmmm/model/diffusion.py`  
**Where:** Update `__init__`, `loss_diffusion`, and the `sample` methods.  
**Action:** Add flags, introduce the training probability ("coin flip"), and pass down predictions iteratively during inference. 

Lines 12-18 (`EquivariantDiffusion.__init__`):
```python
# [EXISTING CODE]
    def __init__(
        self,
        parameterization: EquivariantParameterization,
        diffusion_pos: Optional[ContinuousDiffusion],
        diffusion_h: Optional[ContinuousDiffusion],
# [NEW CODE]
        self_conditioning: bool = False, #self-conditioning: whether to use self-conditioning
        scprop: float = 0.9,  #self-conditioning: probability of using self-conditioning
# [EXISTING CODE]
    ):
        super().__init__()

        self.parameterization = parameterization
        self.diffusions = nn.ModuleDict({"pos": diffusion_pos, "h": diffusion_h})
# [NEW CODE]
        self.self_conditioning = self_conditioning #self-conditioning: whether to use self-conditioning
        self.scprop = scprop #self-conditioning: probability of using self-conditioning
```

Lines ~23-26 (`EquivariantDiffusion.loss_diffusion`):
```python
# [EXISTING CODE]
    def loss_diffusion(self, t: torch.Tensor, batch: Batch | Data):
        latents, targets = self.training_targets(t=t, batch=batch)
# [NEW CODE]
        #self-conditioning: run an inference step without backprop to get previous predictions
        prev_preds = None 
        if self.self_conditioning and torch.rand(1) < self.scprop:
            with torch.no_grad():
                prev_preds = self.parameterization.forward(
                    t=t,
                    **latents,
                    node_index=batch.batch,
                    edge_node_index=batch.edge_node_index,
                )
# [EXISTING CODE]
        preds = self.parameterization.forward(
            t=t,
            **latents,
            node_index=batch.batch,
            edge_node_index=batch.edge_node_index,
# [NEW CODE]
            prev_preds=prev_preds, #self-conditioning: pass previous predictions to the model
# [EXISTING CODE]
        )
```

Lines ~125-140 (`EquivariantDiffusion.sample`):
```python
# [EXISTING CODE]
        if return_traj:
            traj = {
                "pos": [pos_t],
                "h": [h_t],
            }
# [NEW CODE]
        prev_preds = None  #self-conditioning: previous predictions
        
# [EXISTING CODE]
        for i in range(n_steps):
            t = ts[i]
            dt = ts[i + 1] - t

            t = torch.full((num_graphs, 1), t, device=device)

            if method == "em":
# [EXPECT MODIFIED CODE HERE] - Replace what pos_t, h_t unpacks into
                pos_t, h_t, prev_preds = self.reverse_step_em(
                    t=t,
                    dt=dt,
                    pos_t=pos_t,
                    h_t=h_t,
                    node_index=node_index,
                    edge_node_index=edge_node_index,
# [NEW CODE]
                    prev_preds=prev_preds, #self-conditioning: pass previous predictions to the model
# [EXISTING CODE]
                )
```

Lines ~156-173 (`EquivariantDiffusion.reverse_step_em`):
```python
# [EXISTING CODE]
    def reverse_step_em(
        self,
        t: torch.Tensor,
        dt: torch.Tensor,
        pos_t: torch.Tensor,
        h_t: torch.Tensor,
        node_index: torch.Tensor,
        edge_node_index: torch.Tensor,
# [NEW CODE]
        prev_preds: Optional[dict[torch.Tensor, torch.Tensor]] = None #self-conditioning: previous predictions
# [EXISTING CODE]
    ):

        # get NN predictions
        preds = self.parameterization.forward(
            t=t,
            pos=pos_t,
            h=h_t,
            node_index=node_index,
            edge_node_index=edge_node_index,
# [NEW CODE]
            prev_preds=prev_preds, #self-conditioning: pass previous predictions to the model
# [EXISTING CODE]
        )
```

Lines ~185-186 (Return statement of `EquivariantDiffusion.reverse_step_em`):
```python
# [EXISTING CODE]
        return pos_t, h_t
# [NEW CODE COMPILES AS] -> Change the return statement to:
        return pos_t, h_t, preds #self-conditioning: return predictions
```


