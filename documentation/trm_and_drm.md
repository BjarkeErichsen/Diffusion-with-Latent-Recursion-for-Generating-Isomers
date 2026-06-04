# TRM_g3m Latent Recursion Pipeline Implementation Plan

This document details the planned changes needed to incorporate a TRM-like latent recursion structure into the `g3m` model. This runs in addition to (and independently of) the existing self-conditioning mechanism.

## Concept: Latent Recursion and "Z" Matrix Prediction

The core objective is to introduce an additional iterative processing mechanism where the model predicts and refines a global latent scratchpad "Z" alongside standard endpoints.

### 1. The "Z" Matrix Scratchpad
The model will predict an additional latent variable $Z$. At the start, this will be a single latent matrix for the whole graph. $Z$ is used as an input to the message passing layers.

*Architecture of the Forward Pass:*
```python
z_next, x1 = model(xt, t, x1, z_prev)
```
Where `z_prev` is the model's previous prediction of $Z$.

### 2. Control Parameters
To govern this process, we introduce the following structural parameters:
- `latent_recursion` (boolean): Controls whether we execute latent recursion at all (similar to self-conditioning in `train_qm9_sp.yaml`).
- `M x z_dim`: The dimensions of the latent scratchpad. $M$ is an arbitrary size (not necessarily the number of atoms), and $z\_dim$ is the latent dimension size. So $Z \in \mathbb{R}^{M \times z\_dim}$.
- `n` (int, default = 1): **Latent Recursion Steps**. The number of times we predict and update $Z$ *before* denoising/updating $x_1$. Backpropagation occurs through *all* $n$ steps. Note: $x_1$ is *not* updated during these steps; we predict a new version but don't feed it forward. If $n=0$, there is no extra latent recursion, only a single pass predicting both $z_{next}$ and $x_1$.
- `K` (int, default = 2): **Deep Recursion Steps**. The number of times the latent recursions are repeated. $K=2$ implies 1 round of latent recursion without backprop starting from a "none" value of $x_1$, followed by 1 round of latent recursion with backprop starting from the predicted value of $x_1$.

### 3. Training Strategy
During training, we manage both $K$ (Deep Recursion) and $n$ (Latent Recursion).

*Pseudo-code:*
```python
for k in range(K):  # Deep recursion (e.g., K=2, backprop ONLY on the LAST iteration)
    for i in range(n):  # Latent recursion (e.g., n=1, backprop through ALL recursion steps)
        z_next, _ = model(xt, t, x1, z_prev)
        z_prev = z_next
    
    # Final pass to get predicted x1
    z_next, x1 = model(xt, t, x1, z_prev)
    z_prev = z_next
    
    if k < K - 1:  
        # Only backpropagate if we are on the last deep recursion step.
        # Otherwise, detach to prevent tracking gradients through early approximations.
        z_prev = z_next.detach()
        x1 = x1.detach()
```
*Note:* We backpropagate through steps where we update $Z$. This allows the model to learn how to actively "design" $Z$ for better denoising. $x_1$ is not backpropagated through; we only use the deep recursion step to get a strong approximation of how the final $x_1$ will look during inference.

### 4. Inference Strategy
During inference, we don't execute deep recursion ($K$), only latent recursion ($n$). This is because we inherently maintain the most refined history.

*Pseudo-code:*
```python
for i in range(n):  # Latent recursion
    z_next, _ = model(xt, t, x1, z_prev)
    z_prev = z_next

# Final unified pass
z_next, x1 = model(xt, t, x1, z_prev)
z_prev = z_next
```

---








## Architecture and Implementation Details

The following outlines the core areas of the codebase that must be modified to support this design.

### 3.1: Self-Conditioning Implementation Extension
The $x_1$ prediction logic is already partially represented in the model's self-conditioning logic. Currently, there is an "implicit" $K=2$ parameter hardcoded in `model/diffusion.py`.

*Relevant existing code:*
```python
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
```
**ADD TO PLAN:** Modify this code logic to formally include loops matching the training structure above (assuming $K \ge 2$ and $n \ge 0$).

### 3.2: Latent Recursion in Training and Inference
We need to update the forward pass of the diffusion model wrapper to natively track and execute latent recursion steps. This includes wiring the $n$ (latent recursion steps) and $K$ (deep recursion steps) parameters correctly into the core loop.

### 3.3: How do we use z, update z, and initialize z?

#### 1. State & Initialization
* **Dimensions:** Scalar $s \in \mathbb{R}^{N \times F}$, Vector $v \in \mathbb{R}^{N \times F \times 3}$, Latent $Z \in \mathbb{R}^{M \times z\_dim}$.
* **Initialization:** $Z_{base}$ is a static nn.Parameter. At step $n=0$, expand it across the batch dimension to create $Z_{prev}$.

#### 2. The $s$-Only Execution Flow
* **Phase 1: Pre-Message Passing Primer (Layer 0)**
    * **Action:** Nodes read from the global plan before local message passing starts.
    * **Update:** 
        * $s_{new} = s + \text{ZeroInitMLP}(\text{CrossAttn}(Q=s, K=Z_{prev}, V=Z_{prev}))$
        * $v_{new} = v$ (Strict Bypass)
* **Phase 2: The Latent Sync Module (Inside the $L$-layer loop)**
    * *Executes immediately after the local UpdateLayer.*
    * **Read ($Z \leftarrow s$):** 
        * $s$ is converted to a dense representation and $Z$ cross-attends to it.
        * $Z_{new} = Z_{old} + \text{ZeroInitMLP}(\text{CrossAttn}(Q=Z_{old}, K=s_{dense}, V=s_{dense}))$
    * **Compute ($Z \leftrightarrow Z$):**
        * *Standard `nn.TransformerEncoderLayer` with batch_first=True, norm_first=True.*
        * $Z_{proc} = \text{TransformerEncoderLayer}(Z_{new})$
    * **Write ($s \leftarrow Z$):**
        * *Context from $Z$ is written back to the dense representation.*
        * $s_{new\_dense} = s_{dense} + \text{ZeroInitLinear}(\text{CrossAttn}(Q=s_{dense}, K=Z_{proc}, V=Z_{proc}))$
        * $s_{new} = s_{new\_dense}[\text{mask}]$ (convert back to sparse)
        * $v_{new} = v$ (Strict Bypass)
* **Phase 3: Latent Readout**
    * **Action:** Pass the final state back to the diffusion loop for recursion step $n$.
    * **Update:** $Z_{next} = Z_{prev} + \text{ZeroInitLinear}(Z_{proc})$.

---

## 4. The Action Plan
*(Do not write any code yourself, ONLY ADD TO THE PLAN.)*

Below this section, add to the plan, matching the exact style of the implementation plan seen in `self_conditioning_impl_plan.md` (using `[EXISTING CODE]` and `[NEW CODE]` block formatting mapping files and specific line injections). It should reference the code, relevant file, and how you think it should be changed. Here we want to match the description I added.

---

### Phase 1: Configuration Updates

**Target File:** `config/train_qm9_sp.yaml` (Assuming standard hydra layout)  
**Action:** Add the structural parameters to cleanly control the recursion mechanisms and dimension logic.

```yaml
# [NEW CODE] - Add to model arguments
latent_recursion: true
n: 1 # Latent recursion steps
K: 2 # Deep recursion steps
M: 8 # Height of scratchpad
z_dim: 64 # Dimensionality of scratchpad
```
*(Note: Because the intrinsic FlowMol self-conditioning dummy pass evaluates probabilistically, it naturally acts as the first $K=0$ loop, and the main gradient pass naturally acts as $K=1$. Setting K inside the config accurately maps the hyperparameters conceptually as Deep Recursion Steps).*

### Phase 2: Creating the Dedicated Layer Modules

**Target File:** `src_gmmm/nn/layers.py`  
**Action:** Define `LatentSyncModule` matching the exact sequence detailed in Section 3.3. This cleanly encapsulates the dense batching and Read-Compute-Write mechanisms inside the GNN loop constraints, utilizing a `nn.TransformerEncoder` block and completely bypassing the vector track `v`.

```python
# [NEW CODE] - Append to layers.py
from torch_geometric.utils import to_dense_batch
import torch.nn as nn
import torch

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

        # zero init for the mlps both weights and biases
        nn.init.zeros_(self.read_mlp.weight); nn.init.zeros_(self.read_mlp.bias)
        nn.init.zeros_(self.write_mlp.weight); nn.init.zeros_(self.write_mlp.bias)

    def forward(self, z_old: torch.Tensor, s: torch.Tensor, v: torch.Tensor, node_index: torch.Tensor):
        
        #to dense representation
        s_dense, mask = to_dense_batch(s, node_index) # [B, Max_N, node_dim]
        
        # 1 Read:  information s -> z
        o_read, _ = self.read_attn(z_old, s_dense, s_dense, key_padding_mask=~mask)
        z_new = z_old + self.read_mlp(o_read)

        # 2 Compute: z -> z
        z_proc = self.transformer_blocks(z_new)

        # 3 Write:  information z -> s
        o_write, _ = self.write_attn(s_dense, z_proc, z_proc)
        s_new_dense = s_dense + self.write_mlp(o_write) # [B, Max_N, node_dim]

        # back to sparse
        s_new = s_new_dense[mask]

        # TODO: Later add vector processing as well. Currently we just pass v through unchanged.
        v_new = v
        
        return z_proc, s_new, v_new
```

### Phase 3: Updating the Main Encoder Pipeline

**Target File:** `src_gmmm/nn/encoder.py`  
**Where:** `EquivEncoder.__init__` and `EquivEncoder.forward`  
**Action:** Include the `LatentSyncModule` import at the top of the file. Apply the initializations mapped specifically to the PyG batched setup (removing the old primer step since `LatentSyncModule` handles it internally in Phase 2). Execute the Latent modules synchronously with the $L$-layer loop.

```python
# [NEW CODE] - APPEND TO IMPORTS at top of file
from ..nn.layers import LatentSyncModule

# [EXISTING CODE] - __init__
    def __init__(
# [NEW CODE] -> add args latent_recursion=False, M=8, z_dim=64
# ...
        # [NEW CODE] - Initialization inside __init__
        self.latent_recursion = latent_recursion
        if self.latent_recursion:
            # 1: initialization
            self.z_base = nn.Parameter(torch.randn(M, z_dim) * 0.02) #initialize z_base with small random values
             
            # 2: latent recursion / sync layers. 
            self.latent_sync_modules = nn.ModuleList([
                LatentSyncModule(hidden_dim, z_dim) for _ in range(num_layers + 1) #+1 because we need a module as the primer
            ]) 

            # 3: readout residual MLP
            self.z_residual = nn.Linear(z_dim, z_dim)
            nn.init.zeros_(self.z_residual.weight); nn.init.zeros_(self.z_residual.bias) #0 initialization
```

```python
# [EXISTING CODE] - forward signature
    def forward(
# [NEW CODE] -> add argument: z_prev: torch.Tensor = None
# [EXISTING CODE] - forward execution start
        # ... setup t, s, v ...
        
# [NEW CODE] - Latent Initialization & Primer (module 0)
        Z_proc = z_prev
        z_original = z_prev
        if self.latent_recursion:
            # PyG Batching Strategy: Expand dynamically at step n=0
            if z_prev is None:
                num_graphs = node_index.max() + 1
                Z_proc = self.z_base.unsqueeze(0).expand(num_graphs, -1, -1)
                z_original = Z_proc
                
            # Execute Primer Sync (Module 0) before Message Passing
            Z_proc, node_states_s, node_states_v = self.latent_sync_modules[0](
                z_old=Z_proc, s=node_states_s, v=node_states_v, node_index=node_index
            )
            
# [NEW CODE] - Sync modules zipped into the interaction loop (modules 1 to L)
        sync_modules = self.latent_sync_modules[1:] if self.latent_recursion else [None] * len(self.interactions)

# [EXISTING CODE] - Iterative Interaction Loop
        for (
            interaction,
            update, # ... [NEW CODE] and sync_module
            sync_module
        ) in zip(self.interactions, self.updates, sync_modules): 
            node_states_s, node_states_v = interaction.forward(...)
            node_states_s, node_states_v = update(node_states_s, node_states_v)

# [NEW CODE] - Execute Phase 2 mapping immediately after the update layer
            if self.latent_recursion:
                Z_proc, node_states_s, node_states_v = sync_module(
                     z_old=Z_proc, s=node_states_s, v=node_states_v, node_index=node_index
                )

# [NEW CODE] - Readout 
        if self.latent_recursion:
            Z_next = z_original + self.z_residual(Z_proc)
            states["z_next"] = Z_next 
            
        return states
```

### Phase 4: Forward Routing

**Target File:** `src_gmmm/model/score.py`  
**Where:** `EquivariantParameterization.forward`  
**Action:** Pass `z_prev` downward natively.

```python
# [EXISTING CODE]
        edge_node_index: torch.Tensor,
        prev_preds = None #self-conditioning: previous predictions
    ):
# [NEW CODE] -> Add z_prev=None
# ...
        states = self.encoder.forward(
            t=t, h=h, pos=pos, node_index=node_index, edge_node_index=edge_node_index,
            prev_preds=prev_preds,
            z_prev=z_prev # [NEW CODE]
        )
# ...
        # [NEW CODE] - Extract z_next and append to return payload
        if "z_next" in states:
            preds["z"] = states["z_next"]
            
        return preds
```

### Phase 5: The Latent Recursion Loop & Self-Conditioning Integrity

**Target File:** `src_gmmm/model/diffusion.py`  
**Where:** `EquivariantDiffusion.__init__` and `loss_diffusion` (and inherently `sample`)  
**Action:** The explicit directive dictates `prev_preds` (FlowMol conditioning) must remain **completely unmodified/unaltered**. We define a custom independent `for k in range(self.K)` loop to execute Deep Recursion across Latent Recursion tracking our custom predicted starting points `x1`.

```python
# [EXISTING CODE] - __init__
    def __init__(
        # ...
        self_conditioning: bool = False,
        scprop: float = 0.9,
# [NEW CODE]
        latent_recursion: bool = False,
        n: int = 1,
        K: int = 2,
    ):
        # ...
        self.latent_recursion = latent_recursion
        self.n = n
        self.K = K
```

```python
# [EXISTING CODE] - loss_diffusion
        #self-conditioning: run an inference step without backprop to get previous predictions
        prev_preds = None 
        if self.self_conditioning and torch.rand(1) < self.scprop:
            with torch.no_grad():
                prev_preds = self.parameterization.forward(
                    t=t, **latents, node_index=batch.batch, edge_node_index=batch.edge_node_index,
                )


# [NEW CODE COMPILES AS] -> The independent Deep Recursion K loop utilizing prev_preds
        z_prev = None 

        if self.latent_recursion:
            for k in range(self.K): # Explicit K loop specified manually
                
                # 1. Latent Recursion Loop (predicting Z without modifying structural endpoints)
                for i in range(self.n):
                    preds_latent = self.parameterization.forward(
                        t=t, **latents, node_index=batch.batch, edge_node_index=batch.edge_node_index,
                        prev_preds=prev_preds, z_prev=z_prev
                    )
                    z_prev = preds_latent.get("z", None) # Iterate Z matrix
                    
                # 2. Final Unified Pass for this Deep Recursion step
                preds = self.parameterization.forward(
                    t=t, **latents, node_index=batch.batch, edge_node_index=batch.edge_node_index,
                    prev_preds=prev_preds, z_prev=z_prev
                )
                z_prev = preds.get("z", None)
                
                # 3. Update structure for the next Deep Recursion step
                if k < self.K - 1:
                    prev_preds = {"pos": preds["pos"].detach(), "h": preds["h"].detach()}
                    if z_prev is not None: z_prev = z_prev.detach()
                    
        else:
            # Standard single pass if latent recursion disabled
            preds = self.parameterization.forward(
                t=t, **latents, node_index=batch.batch, edge_node_index=batch.edge_node_index,
                prev_preds=prev_preds
            )
        
        # End loop, preds holds the final target predictions for loss computation
```

### Phase 6: Inference & Sampling Extension

**Target File:** `src_gmmm/model/diffusion.py`  
**Where:** `EquivariantDiffusion.reverse_step_em`  
**Action:** Extract the `z` history intrinsically carried in `prev_preds` from the previous diffusion timestep, and execute the inner `n` latent recursion loop to refine `z` before making the main prediction step. Keep the implementation minimal by only using the inner latent recursion loop (no $K$ deep recursion loop is used during sampling).

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
        prev_preds: Optional[dict[torch.Tensor, torch.Tensor]] = None #self-conditioning: previous predictions
    ):

# [NEW CODE] - Inner n loop for inference
        z_prev = prev_preds.get("z", None) if prev_preds is not None else None
        
        if getattr(self, "latent_recursion", False):
            for i in range(self.n):
                preds_latent = self.parameterization.forward(
                    t=t, pos=pos_t, h=h_t, node_index=node_index, edge_node_index=edge_node_index,
                    prev_preds=prev_preds, z_prev=z_prev
                )
                z_prev = preds_latent.get("z", None)
                
# [EXISTING CODE]
        # get NN predictions
        preds = self.parameterization.forward(
            t=t,
            pos=pos_t,
            h=h_t,
            node_index=node_index,
            edge_node_index=edge_node_index,
            prev_preds=prev_preds, #self-conditioning: pass previous predictions to the model
# [NEW CODE] -> Add z_prev argument
            z_prev=z_prev
        )
# ... continues unaltered ...
```

### Phase 7: TRM Evaluation Tracking (`eval_trm`)

**Target File 1:** `src_gmmm/utils/trm_eval.py`  
**Action:** Create a single function `save_trm_eval_data` using `HydraConfig` to route logs to the active run folder.

```python
def save_trm_eval_data(data: Dict[str, Any]):
    # Uses HydraConfig.get().runtime.output_dir
    # Saves unified lists for X_t, X_1, and transposed iteration lists Z_i.pt
```

**Target File 2:** `src_gmmm/model/diffusion.py`  
**Action:** Use `eval_trm_data` dictionary to buffer states and call `save_trm_eval_data` at the end of `sample()`.


**Target File 2:** `src_gmmm/model/diffusion.py`  
**Where:** `EquivariantDiffusion.__init__`, `reverse_step_em`, and `sample`  
**Action:** Accumulate lists internally inside a dictionary, then call `save_trm_eval_data(eval_trm_data)` at the end.

```python
# [EXISTING CODE] - __init__
# ...
        latent_recursion: bool = False,
        n: int = 1,
        K: int = 2,
# [NEW CODE] -> add eval_trm parameter
        eval_trm: bool = False,
    ):
        # ...
        self.K = K
        self.eval_trm = eval_trm
```

```python
# [EXISTING CODE] - reverse_step_em
    def reverse_step_em(
        self,
        # ...
        prev_preds: Optional[dict[torch.Tensor, torch.Tensor]] = None 
    ):
# [NEW CODE] - Init tracking sequence
        z_intermediates = []
        
        z_prev = prev_preds.get("z", None) if prev_preds is not None else None
        
        if self.latent_recursion:
            for i in range(self.n):
                preds = self.parameterization.forward(
                    # ...
                )
                z_prev = preds.get("z", None)
                if self.eval_trm and z_prev is not None:
                    z_intermediates.append(z_prev.detach().cpu())

        # get NN predictions
        preds = self.parameterization.forward(
        # ...
        
# [NEW CODE] - Pass tracked tensors into payload for Sample step
        if self.eval_trm and "z" in preds:
            z_intermediates.append(preds["z"].detach().cpu())
            preds["z_intermediates"] = z_intermediates 

        # reverse step on each modality
# ...
```

```python
# [EXISTING CODE] - sample 
        prev_preds = None  
# [NEW CODE] - Setup eval tracking buffers dict
# [DEPRECATED - Moved tracking logic inside loop natively to `if i == 0` check]

# [EXISTING CODE] - Inside the n_steps loop
        for i in range(n_steps):
            # ...
            if method == "em":
                pos_t, h_t, prev_preds = self.reverse_step_em( 
                    # ...
                )

# [NEW CODE] - Track variables dynamically without assuming categorical 'h' is tracked
            if self.eval_trm:
                if i == 0: 
                    eval_trm_data = {
                        "X_t": {"pos": [], "h": []}, 
                        "X_1": {"pos": [], "h": []}, 
                        "Z": [],
                        "val_datapoint": {"pos": to_dense_batch(batch.pos, batch.batch)[0].detach().cpu(), 
                                          "h": to_dense_batch(batch.h, batch.batch)[0].detach().cpu(), 
                                          "edge_node_index": batch.edge_node_index.detach().cpu(),
                                          "mask": to_dense_batch(batch.pos, batch.batch)[1].detach().cpu()}
                    }  #evaluation trm

                if pos_t is not None:
                    eval_trm_data["X_t"]["pos"].append(to_dense_batch(pos_t, node_index)[0].detach().cpu())
                
                if h_t is not None:
                    eval_trm_data["X_t"]["h"].append(to_dense_batch(h_t, node_index)[0].detach().cpu())
                    
                if prev_preds:
                    if "pos" in prev_preds and prev_preds["pos"] is not None:
                        eval_trm_data["X_1"]["pos"].append(to_dense_batch(prev_preds["pos"], node_index)[0].detach().cpu())
                    if "h" in prev_preds and prev_preds["h"] is not None:
                        eval_trm_data["X_1"]["h"].append(to_dense_batch(prev_preds["h"], node_index)[0].detach().cpu())
                    
                    eval_trm_data["Z"].append(prev_preds.get("z_intermediates", []))
                
                if i == n_steps - 1:
                    save_trm_eval_data(eval_trm_data)
                     
# [EXISTING CODE] - End of sample function
        samples = {
            "pos": pos_t,
            "h": h_t,
        }

# [NEW CODE] - Serialize tracked elements to disk efficiently via function call
# [DEPRECATED - serialization moved to `if i == n_steps - 1` inside main loop]

# [EXISTING CODE]
        if return_traj:
            return samples, traj
```

### Phase 8: Add Latent Normalization (RMSNorm)

**Target File:** `src_gmmm/nn/layers.py`  
**Location:** `LatentSyncModule`  
**Action:** Implement and incorporate a full RMS Normalization layer with learnable weights. The implementation specifically focuses on normalizing the **residual updates** and the **final output** to ensure a clean identity path while maintaining stability.

**Specific Logic:**
*   **Layer Definition:** Define an `RMSNorm` module that calculates $\frac{x}{\text{RMS}(x)} * \gamma$, where $\gamma$ is a learnable parameter of size `z_dim`.
*   **Normalization of Residual (Read):** Pass the result of the `read_mlp` projection through `self.norm_read` **before** adding it to the parent state. This keeps the identity path $z_{old} + \dots$ clean while ensuring the incoming information is scale-standardized.
*   **Normalization of Final State (Compute):** Pass the output of the transformer blocks through `self.norm_compute`. This provides the essential global magnitude control for $Z$ before it is used for writing to nodes or passed to the next recursion step.

```python
# [NEW CODE] - Actual Logic in LatentSyncModule.forward
# 1. Read: Normalize only the update to keep the identity path intact
o_read_norm = self.norm_read(self.read_mlp(o_read)) 
z_new = z_old + o_read_norm #residual connection is kept intact -> no completicated identity path

# 2. Compute: Normalize the final output of the transformer
if self.skip_transformer_block:
    z_proc = z_new
else:
    z_proc = self.transformer_blocks(z_new)

#Normalize output of transformer #TODO maybe not needed
z_proc = self.norm_compute(z_proc)
```


### Phase 9: (not done yet) Adding time conditioning to Z

### Phase 10: (not done yet) Masking Padded Atoms in Latent Write Step 

### Phase 11: (not done yet)  Added pos embeddings


file: trm_utilities is added

self.z_pe = SinusoidalPositionalEncoding(d_model=z_dim, max_len=M)


### Phase 12: (baseline parameterizations) 


### Phase 13: Use s instead of z in the TRM-g3m model
CRITICAL: do not implement any code, only give a plan with the exact code i should add similar to the previous phases.

We want to transfer the s representation learned in the previous step to the next step. This REPLACES the current z updates in the model, i.e. 
`latent_states = s`.

Currently, we will only use the s_t-1 representation to (help) initialize the s reprsentation of the next layer iteration (s_t) 

As this replaces the current z-updates, we will use the same meta variables, fx we will replace the layer that currently controls z updates with a layer that controls s updates and uses it for the same purpose as z is currently used. 
#for good order just comment out current z -specific code. 

In short, make as few changes as possible, replacing z with s_prev, and construct a simple linearl ayer that helps the previous s be usefull for initializing the next s. 

Dont implement any fancyness like cross attention as was done with z, also this will still be called latent-recursion and be controlled by the same variables in the config.

**Target File:** `src_gmmm/nn/encoder.py`  
**Where:** `EquivEncoder.__init__` and `EquivEncoder.forward`  
**Action:** Replace `z`-specific logic with `s_prev` logic. Comment out `z` initialization and `LatentSyncModule` instantiations. Introduce a simple linear layer `self.s_sync` to process `s_prev` (which is passed down using the existing `z_prev` argument). In the `forward` pass, apply `self.s_sync(z_prev)` to `node_states_s`. Finally, return `node_states_s` under the `z` key so that it seamlessly loops back into the next iteration as `z_prev`.

```python
# [EXISTING CODE] - __init__
        #latent recursion
        self.latent_recursion = latent_recursion or latent_recursion_v2
        if self.latent_recursion:
            # 1: initialization
# [NEW CODE] - Comment out z-specific initialization and add s_sync
            # self.z_base = nn.Parameter(torch.randn(M, z_dim)* 0.02) #initialize z_base with small random values
             
            # 2: latent recursion / sync layers. 
            # self.latent_sync_modules = nn.ModuleList([
            #     LatentSyncModule(hidden_dim, z_dim) for _ in range(num_layers + 1) #+1 because we need a module as the primer
            # ]) 

            # 3: readout residual MLP
            # self.z_residual = nn.Linear(z_dim, z_dim)
            # nn.init.zeros_(self.z_residual.weight); nn.init.zeros_(self.z_residual.bias) #0 initialization

            # self.z_pe = SinusoidalPositionalEncoding(d_model=z_dim, max_len=M)
            
            # Simple linear layer to help the previous s be useful for initializing the next s
            self.s_sync = nn.Linear(hidden_dim, hidden_dim)
```

```python
# [EXISTING CODE] - forward Execution Start
        #latent recursion: initialize z_prev and run primer
        Z_proc = z_prev
        z_original = z_prev
        if self.latent_recursion:
# [NEW CODE] - Comment out z primer, inject s_prev directly into node_states_s
            # if z_prev is None:
            #     num_graphs = node_index.max() + 1
            #     Z_proc = self.z_base.unsqueeze(0).expand(num_graphs, -1, -1)
            #
            #     Z_proc = Z_proc + self.z_pe() #positional encoding
            #
            #     z_original = Z_proc
            # 
            # #primer (using module 1)
            # Z_proc, node_states_s, node_states_v = self.latent_sync_modules[0](
            #     z_old = Z_proc,
            #     s = node_states_s,
            #     v = node_states_v,
            #     node_index = node_index
            #     )

            # z_prev is now acting as s_prev
            if z_prev is not None:
                node_states_s = node_states_s + self.s_sync(z_prev)
                
# [EXISTING CODE] - Iterative Interaction Loop
        # sync_modules = self.latent_sync_modules[1:] if self.latent_recursion else [None] * len(self.interactions) # latent recursion:we need this, otherwise running without latent recursion will throw an error

# [NEW CODE] - Standard loop (removing sync_modules from zip)
        for (
            interaction,
            update
        ) in zip(self.interactions, self.updates):
# [EXISTING CODE]
            node_states_s, node_states_v = interaction.forward(
                node_states_s=node_states_s,
                node_states_v=node_states_v,
                edge_states=edge_states,
                unit_vectors=unit_vectors,
                node_index=node_index,
                edge_node_index=edge_node_index,
            )
            node_states_s, node_states_v = update(node_states_s, node_states_v)
            
# [NEW CODE] - Comment out sync_module call inside loop
            # if False: # self.latent_recursion: #: #Set to false to disable
            #     Z_proc, node_states_s, node_states_v = sync_module(
            #         z_old = Z_proc,
            #         s = node_states_s,
            #         v = node_states_v,
            #         node_index = node_index
            #     )
        
        states = {"s": node_states_s, "v": node_states_v}
        
# [EXISTING CODE] - Readout
        #latent recursion: final readout with residual connection
        if self.latent_recursion:
# [NEW CODE] - Return current s to be used as s_prev (z_prev) in the next iteration
            # Z_next = z_original + self.z_residual(Z_proc)
            # states["z"] = Z_next
            
            # We replace z with s_prev. Outputting it as "z" routes it seamlessly through diffusion.py
            states["z"] = node_states_s 

        return states
```

### Phase 14: Removal of Legacy Tracking & Code Cleanup
**Target Files:** `configs/train_qm9_sp.yaml`, `src_gmmm/nn/encoder.py`, `src_gmmm/model/diffusion.py`, `src_gmmm/utils/trm_eval.py`  
**Action:** Before implementing the new tracking mechanism, clean up the codebase to remove deprecated features.
- **Remove `eval_trm`**: Strip out all config parameters, model logic, and callback logic related to `eval_trm`.
- **Remove `latent_recursion_v2`**: Remove `latent_recursion_v2` from `configs/train_qm9_sp.yaml`, `src_gmmm/nn/encoder.py`, and any other locations. Remove unneeded parameter initializations (e.g. inside the old `latent_recursion_v2` blocks).

### Phase 15: Design Options for Scalable Internal Representation Tracking
Tracking internal representations efficiently without suffocating disk space or slowing down training is a common challenge in deep learning research. Below are 3 designs ranging from basic to highly professional/scalable.

#### Level 1: Config-Driven Manual Context (Basic Professionalism)
*Best for: Rapid prototyping, single-node training, small-to-medium graphs.*

**How it works:**
Introduce a `Tracker` singleton or context object passed down through the model. In the config, specify exactly which variables to track (e.g., `track_vars: ["node_states_s", "edge_states"]`) and a sampling frequency (e.g., `track_frequency: "val_first_batch"`).
In the model code, manually add hooks: 
```python
if self.tracker.should_track("node_states_s"):
    self.tracker.log("node_states_s", node_states_s)
```
**Storage:** 
The `Tracker` aggregates these tensors in RAM until the batch/trajectory ends, then dumps them using `torch.save()` into an organized hierarchy:
`output_dir/representations/epoch_005/val_batch_0/node_states_s.pt`

**Pros:** Extremely simple to implement; highly transparent; easy to debug.
**Cons:** Clutters model code with `if` statements; synchronous disk I/O can cause minor training stalls.

#### Level 2: PyTorch Forward Hooks & Lightning Callbacks (Intermediate Scalability)
*Best for: Clean model code, complex architectures, standard research workflows.*

**How it works:**
Instead of modifying the core model logic, use PyTorch's native `register_forward_hook`. Create a PyTorch Lightning Callback called `RepresentationTrackerCallback`. 
In the config, map module names to the variables you want:
```yaml
tracking:
  frequency: "val_every_5_epochs"
  modules:
    "encoder.interactions.0": ["output"]
    "encoder.sc_layer": ["input", "output"]
```
The callback attaches hooks to these specific modules *only* when the frequency condition is met. 

**Storage:** 
When the hook fires, it moves the data to CPU asynchronously and queues it to a background thread. The background thread saves the data into HDF5 (`.h5`) format or chunked `.pt` files inside an organized representations folder.

**Pros:** Zero modification to core model forward passes (perfect separation of concerns); no overhead when not tracking; HDF5 is highly structured.
**Cons:** HDF5 can sometimes be tricky with variable-sized PyG graphs; attaching/detaching hooks requires careful lifecycle management.

#### Level 3: Asynchronous Telemetry Probes with Chunked Zarr/WebDataset (High Professionalism/MLOps)
*Best for: Massive scale, multi-node distributed training, terabytes of representation data.*

**How it works:**
Implement a Publisher-Subscriber telemetry system. The model emits lightweight signals: `Probe.emit("encoder/node_s", tensor)`. 
A dedicated background daemon (running on a separate thread or process) subscribes to these probes. If the current batch isn't targeted for saving, the `emit` is a no-op (near zero overhead). 

**Storage:**
The daemon writes data asynchronously using **Zarr** or **WebDataset** formats. The directory structure operates like a database:
`output_dir/telemetry.zarr/epochs/005/encoder/node_s/...`

**Pros:** Completely asynchronous (never blocks the GPU); scales to infinite data sizes; standard format used by data engineers; highly optimized parallel reads.
**Cons:** Significant engineering effort up front; overkill if you only want to look at a few tensors occasionally.

### Phase 16: Level 2 Tracker Implementation (Elaborated)

To make PyTorch's `register_forward_hook` robust and avoid the complexities of inspecting internal `kwargs` or tuple outputs of existing layers (especially when dealing with PyG's `node_index`), we introduce an explicit `RepresentationProbe` layer. This is essentially an identity layer that acts as a clean anchor point for our hooks, guaranteeing we have access to both the features and the graph topology (`node_index`).

**Target File:** `src_gmmm/nn/layers.py` (or directly in `encoder.py`)  
**Action:** Define the `RepresentationProbe` which passes data through unmodified.

```python
# [NEW CODE] - Add Probe Definition
import torch.nn as nn

class RepresentationProbe(nn.Module):
    """
    Dummy layer used purely as an anchor for PyTorch forward hooks.
    Takes the tensor and its batch index, returning the tensor unmodified.
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, x, node_index=None):
        return x
```

**Target File:** `src_gmmm/nn/encoder.py`  
**Where:** `EquivEncoder.__init__` and `EquivEncoder.forward`  
**Action:** Instantiate probes for the specific locations you wish to track, and insert them into the forward pass.

```python
# [EXISTING CODE] - __init__
# ...
        #update edge states
        self.update_edge_states = update_edge_states
        if update_edge_states:
            self.edge_update_layer = EdgeUpdateLayer(self.hidden_dim, self.edge_embedding.out_features, architecture="layernorm")

# [NEW CODE] - Add Tracking Probes
        self.probes = nn.ModuleDict({
            "post_sc_s": RepresentationProbe(),
            "post_sc_v": RepresentationProbe(),
            "layer_0_s": RepresentationProbe(),
            "layer_0_v": RepresentationProbe(),
            # Add more as needed...
        })
```

```python
# [EXISTING CODE] - forward
        #self-conditioning: update node and edge states with previous predictions. Done BEFORE interaction and update layers.
        if self.self_conditioning and prev_preds is not None:
            # ... sc_layer forward ...
            if not self.ablations.get("ablate_v", False):
                # ...
                node_states_v = self.v_mlp(v)
                
# [NEW CODE] - Explicitly Probe intermediate states
            node_states_s = self.probes["post_sc_s"](node_states_s, node_index)
            node_states_v = self.probes["post_sc_v"](node_states_v, node_index)

# [EXISTING CODE] - Interaction Loop
        for i, (
            interaction,
            update,
        ) in enumerate(zip(self.interactions, self.updates)):
            node_states_s, node_states_v = interaction.forward(
                # ...
            )
            node_states_s, node_states_v = update(node_states_s, node_states_v)
            
            if self.update_edge_states:
                edge_states = self.edge_update_layer(edge_states, node_states_s, node_states_v, edge_node_index)

# [NEW CODE] - Probe inside the loop (e.g., just for layer 0)
            if i == 0:
                node_states_s = self.probes["layer_0_s"](node_states_s, node_index)
                node_states_v = self.probes["layer_0_v"](node_states_v, node_index)
```

**Target File:** `src_gmmm/utils/tracking.py`  
**Action:** Create the `RepresentationTrackerCallback` to attach to these explicit probes. Because the probe receives `(x, node_index)`, we can optionally reshape it to a dense batch before saving, making it much easier to analyze later.

```python
# [NEW CODE] - Create new file src_gmmm/utils/tracking.py
import torch
import pytorch_lightning as pl
import threading
import queue
import os
from torch_geometric.utils import to_dense_batch

class RepresentationTrackerCallback(pl.Callback):
    def __init__(self, frequency: str = "none", modules: dict = None):
        super().__init__()
        self.frequency = frequency
        self.modules_cfg = modules or {}
        self.save_queue = queue.Queue()
        self.hooks = []
        self.epoch = 0
        self.batch_idx = 0
        
        self.saver_thread = threading.Thread(target=self._save_worker, daemon=True)
        self.saver_thread.start()

    def _save_worker(self):
        while True:
            item = self.save_queue.get()
            if item is None: break
            filepath, data = item
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            torch.save(data, filepath)
            self.save_queue.task_done()
            
    def _get_hook(self, module_name):
        # We use with_kwargs=True if PyTorch >= 2.0, but to be safe for older versions,
        # we assume Probe receives (x, node_index) as positional args if passed that way,
        # or we just grab the output.
        def hook(module, inputs, output):
            x = output.detach().cpu()
            
            # If node_index was passed as the second positional argument to the probe:
            if len(inputs) > 1 and inputs[1] is not None:
                node_index = inputs[1].detach().cpu()
                # Optional: Convert to dense batch so it's [Batch_Size, Max_Nodes, Features]
                # data, mask = to_dense_batch(x, node_index)
                data = {"x": x, "node_index": node_index}
            else:
                data = x
                
            filepath = os.path.join(self.output_dir, f"epoch_{self.epoch:03d}", f"batch_{self.batch_idx}", f"{module_name}.pt")
            self.save_queue.put((filepath, data))
        return hook

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        self.epoch = trainer.current_epoch
        self.batch_idx = batch_idx
        log_dir = trainer.logger.save_dir if trainer.logger else "."
        name = trainer.logger.name if trainer.logger and trainer.logger.name else "run"
        self.output_dir = os.path.join(log_dir, name, "representations")
        
        if self.frequency == "val_first_batch" and batch_idx == 0:
            should_track = True
        elif self.frequency.startswith("val_every_") and "epochs" in self.frequency:
            n = int(self.frequency.split("_")[2])
            should_track = (self.epoch % n == 0) and (batch_idx == 0)
        else:
            should_track = False

        if should_track:
            for name, module in pl_module.named_modules():
                # E.g., name might be "model.parameterization.encoder.probes.post_sc_s"
                if any(k in name for k in self.modules_cfg):
                    # Clean the filename up based on the probe name
                    clean_name = name.split(".")[-1] 
                    h = module.register_forward_hook(self._get_hook(clean_name))
                    self.hooks.append(h)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def on_train_end(self, trainer, pl_module):
        self.save_queue.put(None)
        self.saver_thread.join()
```

**Target File:** `configs/train_qm9_sp.yaml`  
**Action:** Map the tracker to our explicit probes.

```yaml
# [EXISTING CODE]
callbacks:
  model_checkpoint:
    # ... 
    
# [NEW CODE] - Append to callbacks
  representation_tracker:
    _target_: src_gmmm.utils.tracking.RepresentationTrackerCallback
    frequency: "val_every_5_epochs" 
    modules:
      - "probes.post_sc_s"
      - "probes.layer_0_s"
```











# Denoising Recursion Model (DRM) Implementation Overview

This document provides a detailed breakdown of the fundamental elements of the Denoising Recursion Model (DRM), expanding on the internal mechanics of the denoising process, the handling of timesteps, and the exact flow of latent states.

## 1. Denoising Mechanism (Hierarchical Recursion)
The fundamental engine of the DRM is not a simple linear loop, but a **hierarchical recurrent reasoning module** (defined in `models/recursive_reasoning/rm.py`). The model maintains two distinct latent states:
1.  **`decode_latent`**: Represents the evolving solution grid (the target answer).
2.  **`scratchpad_latent`**: A hidden workspace for intermediate computation.

*   **The Unrolling Loops**:
    The model unrolls its shared `RecurrentBlock` (a stack of Transformer/SwiGLU layers) in a nested loop defined by `H_cycles` (outer) and `L_cycles` / `grad_cycles` (inner).
    *   **Inner Loop (Scratchpad Updates)**: For `L_cycles` iterations, the `scratchpad_latent` is updated by processing the current `decode_latent` and the fixed `input_embeddings`.
    *   **Outer Loop (Decode Update)**: After the `L` scratchpad updates, the `decode_latent` is updated once by processing the finalized `scratchpad_latent`.
*   **The Gradient Window ($k$ steps)**:
    To manage memory and training stability, the model employs Truncated Backpropagation Through Time (TBPTT). 
    *   The first `H_cycles - 1` outer loops are run inside a `torch.no_grad()` block (referred to as `warmup_cycles`).
    *   Only the **final outer loop** (the last $L$ scratchpad updates and 1 decode update) is recorded with gradients. This final window is the "$k$ recursive denoising steps" supervised during training.
    *   **Inference Behavior**: During inference, the model still executes the full $H \times L$ sequence of updates. The "window" concept is purely a training optimization for memory and gradient flow.

## 2. Timestep Embeddings
The DRM distinguishes between the inner recursion steps and the outer diffusion steps. A fundamental design choice of this model is that **time is entirely implicit**.

*   **Within the Recursion Window ($k$ / $H, L$ cycles)**:
    *   The model is **strictly time-invariant**. The `RecurrentBlock` applies identical weights at every step.
    *   There is no explicit step counter, positional time encoding, or scalar injected into the hidden states to tell the model "which step" it is on. The progression of time is encoded purely in the evolving data within the `scratchpad_latent` and `decode_latent`.
*   **Overall Denoising / Diffusion Timestep ($\tau$ or $i \in T$)**:
    *   $\tau$ controls the **masking ratio** of the target grid (via `DiscreteScheduler` in `diffusion/schedulers.py`).
    *   While the codebase contains a `_time_embedding` function designed to generate standard sinusoidal diffusion time encodings, **it is explicitly unused**.
    *   The model is forced to be "noise-agnostic." It must infer the corruption level simply by observing the density of `MASK` tokens in the `decode_latent` and apply its universal denoising rules accordingly.

## 3. Flow of States and Information
The flow of information dictates how the problem input ($X$) and the noisy target ($Y$) are resolved into a clean output.

### A. Initialization
1.  **Input ($X$)**: The problem grid is embedded into fixed `input_embeddings`.
2.  **Scratchpad**: Initialized from a learned static buffer (`scratchpad_init`).
3.  **Decode Target**: 
    *   *(Train)*: $Y_{target}$ is corrupted by masking a fraction of tokens based on a random $\tau$. This $Y_{noisy}$ is embedded to become the initial `decode_latent`.
    *   *(Inference)*: Initialized as a fully masked grid ($Y_{fully\_masked}$).

### B. Forward Process (Train & Inference Step)
The core unrolling executes as follows:
```text
Loop H_cycles times:
    Loop L_cycles times:
        scratchpad_latent = Block(scratchpad_latent, decode_latent + input_embeddings)
    decode_latent = Block(decode_latent, scratchpad_latent)
```

**Notation:**
*   **$X$**: `input_embeddings` (The fixed problem/puzzle constraints).
*   **$Y$**: `decode_latent` (The evolving target solution/grid).
*   **$H$**: `scratchpad_latent` (The hidden workspace/reasoning memory).

*   **Decoding**: The final `decode_latent` is passed through an `lm_head` to predict token logits (the clean grid $\hat{Y}$).
*   *(Train only)*: Cross-entropy loss is applied between the logits and the uncorrupted $Y_{target}$.

### C. Iterative Denoising Flow (Inference Only)
During evaluation, the model runs the forward process $T$ times (defined by `diffusion_num_inference_steps`), chaining the states together:
1.  **Predict**: Run the full $H \times L$ forward process to predict a completely filled grid $\hat{Y}_i$.
2.  **Carry Over**: The finalized `scratchpad_latent` is **kept and passed** to the next timestep $i+1$, providing continuous memory across the denoising trajectory.
3.  **Remask**: Based on the schedule for step $i+1$ (a lower noise ratio), a subset of the predicted tokens in $\hat{Y}_i$ are replaced with `MASK` tokens, generating $Y_{noisy, i+1}$.
4.  **Re-embed**: This newly masked grid is embedded to form the fresh `decode_latent` for the next forward pass.
5.  **Repeat**: This continues until $\tau = 0$ (no masks remaining).

## 4. Specific Training vs. Inference Mechanics

### A. Encoding the Step in the Window
*   **Is the step encoded?** No. The progression of the loop (e.g., whether it is step 1 or step $L$ in the inner loop) is **not encoded** as a feature. The identical transformer block is applied recursively. 
*   **Is it used during training (different from $\tau$)?** Because it is not encoded, the model does not explicitly "know" the difference between step 1 and step $L$. It differs fundamentally from $\tau$ in that $\tau$ determines the physical input state (how many `MASK` tokens are present) at the start of the window, whereas the window step is merely an internal unit of computation.

### B. Re-masking During Training
*   **Do we re-mask during training?** **No.** Re-masking is exclusively an inference-time mechanism used to bridge the gap between outer diffusion steps.
*   **Training Behavior**: During training, a single random diffusion timestep $\tau$ is sampled per batch. The target grid $Y_{target}$ is masked once based on $\tau$ to create the initial `decode_latent`. The model then unrolls its full forward pass ($H \times L$ loops) operating entirely in the continuous latent space. It does not map back to discrete tokens and does not apply new masks midway through the training forward pass.

### C. Decoding into $Y$ Directly
*   **Do we apply the decoder to decode into $Y$ directly?** **Yes.** 
*   **Implementation**: At the very end of the $H \times L$ recursive loops, the final `decode_latent` state is passed directly through a linear layer (`self.lm_head` in `rm.py`). This head projects the latent vectors directly into vocabulary logits representing the predicted clean grid $\hat{Y}$. 
*   **Supervision**: These predicted logits are then directly compared against the fully uncorrupted ground truth tokens ($Y_{target}$) using a standard Cross-Entropy loss.

## 5. Architecture and State Evolution

This section details the explicit shapes and mathematical transfer equations for the latents as they move through the architecture.

### Shape Variables
*   **$B$**: Batch size.
*   **$S$**: Sequence length. This is typically the flattened grid size $M$ (e.g., $30 \times 30 = 900$) plus the `puzzle_emb_len` (e.g., $16$), yielding $S = 916$.
*   **$D$**: Hidden dimension (`hidden_size`).
*   **$V$**: Vocabulary size (number of possible cell colors/states).

Both primary latents—`scratchpad_latent` ($Z$) and `decode_latent` ($Y$)—maintain a constant shape of **`[B, S, D]`** throughout the entire forward process. The input sequence $X$ is also embedded into this shape.

### Forward Pass Equations (Inner Recursion)
During a single outer diffusion timestep $i$, the model updates its latents using the shared `RecurrentBlock` ($\Phi$). Let the superscript $(i, j)$ denote the state at diffusion step $i$ and inner-loop update $j$. 

For each iteration $j$:
1.  **Scratchpad Update**: The scratchpad absorbs the current target hypothesis and the fixed input constraints.
    $$Z^{(i, j+1)} = \Phi(Z^{(i, j)}, \; Y^{(i, j)} + X_{embed})$$
2.  **Decode Update**: The target hypothesis is refined using the updated scratchpad.
    $$Y^{(i, j+1)} = \Phi(Y^{(i, j)}, \; Z^{(i, j+1)})$$

### Carry Over Equations (Outer Denoising)
At the end of the recursion window for diffusion timestep $i$ (let's denote the final inner step as $k$), we obtain the finalized latents $Z^{(i, k)}$ and $Y^{(i, k)}$. The model generates discrete predictions via the Linear Language Model Head:
$$\hat{Y}^{(i)} = \text{argmax}(\text{LM\_Head}(Y^{(i, k)})) \quad \text{Shape: } [B, S]$$

When moving to the next inference timestep $i+1$, the two latents are handled fundamentally differently:

1.  **Scratchpad ($Z$) Carry Over**:
    The scratchpad is passed forward perfectly untouched, providing continuous working memory.
    $$Z^{(i+1, 0)} = Z^{(i, k)}$$

2.  **Decode Latent ($Y$) Carry Over**:
    The continuous decode representation is discarded. Instead, the discrete predictions $\hat{Y}^{(i)}$ are corrupted based on the next noise level $\tau_{i+1}$ and re-embedded from scratch.
    $$Y_{noisy}^{(i+1)} = \text{Remask}(\hat{Y}^{(i)}, \; \tau_{i+1})$$
    $$Y^{(i+1, 0)} = \text{Embed}(Y_{noisy}^{(i+1)})$$
    *(Note: The `Remask` function preserves high-confidence predictions while injecting `MASK` tokens back into uncertain locations).*

## 6. DRM Inference Pseudo-code (Advanced)

**Notation:** $X$ = Input, $Y$ = Decode Latent, $H$ = Scratchpad Latent.

*Note: During inference, the full $H \times L$ recursion depth is executed at every diffusion timestep $t$. The "gradient window" is a training-only optimization for memory.*

The following pseudo-code describes the iterative denoising update during evaluation (inference), mapping the discrete re-masking logic back into the continuous latent space.

```python
# --- INITIALIZATION ---
t_steps = scheduler.get_inference_timesteps()
carry = model.initial_carry(batch) # Fully masked decode_latent
puzzle_prefix = carry.decode_latent[:, :puzzle_len].clone()

# --- DENOISING LOOP ---
for t in t_steps:
    # --- 1. INTERNAL RECURSION (The Depth Window) ---
    # For every diffusion step t, we run the full H x L recursion.
    for h in range(H_cycles):
        for l in range(L_cycles):
            # Update Scratchpad (H) using current Decode (Y) and Input (X)
            carry.H = Block(carry.H, carry.Y + batch.X)
        
        # Update Decode (Y) using the finalized Scratchpad (H)
        carry.Y = Block(carry.Y, carry.H)

    # Decode continuous Y into discrete tokens for re-masking
    logits = model.lm_head(carry.Y)
    grid = argmax(logits) 
    p_mask = t / total_train_timesteps

    # 2. Denoising Update (Remasking)
    if confidence_masking:
        # Calculate how many tokens to 'forget' at this noise level
        n_mask = int(p_mask * count_nonzero(grid))
        
        # Identity tokens model is least sure about
        conf = softmax(logits).max(dim=-1)
        
        # Select 99% by lowest confidence, 1% random for exploration
        target_indices = select_top_k(1.0 - conf, k=0.99 * n_mask)
        target_indices += select_random(remaining, k=0.01 * n_mask)
        
        grid[target_indices] = MASK_TOKEN_ID
    else:
        # Simple uniform noise destruction
        grid[random_uniform(shape(grid)) < p_mask] = MASK_TOKEN_ID

    # 3. State Injection for next timestep
    # We discard the continuous decode_latent but KEEP the scratchpad_latent
    new_decode_latent = concat([puzzle_prefix, embed(grid)], dim=1)
    
    carry.recurrent_state = RecurrentState(
        decode_latent=new_decode_latent,
        scratchpad_latent=carry.recurrent_state.scratchpad_latent # Continuous memory
    )
```

