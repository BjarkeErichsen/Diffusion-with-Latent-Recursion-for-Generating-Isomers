# TRM_g3m Architecture & Methodology Overview

## 1. Data Input Representation
TRM_g3m operates exclusively on geometric graphs representing physical systems (e.g., molecules). It specifically delineates between spatial coordinates and continuous attribute data. 

**Core Inputs:**
*   **`pos` ($\mathbb{R}^{N \times 3}$):** Explicit 3D Euclidean coordinates of the nodes (atoms). This track is physically rotated and translated, mandating equivariant operations to preserve system constraints.
*   **`h` ($\mathbb{R}^{N \times C}$ or $\mathbb{Z}^{N}$):** The node features, almost exclusively representing atomic species (e.g., Carbon, Hydrogen) or formal charges. TRM_g3m allows `h` to be passed as rigid discrete categorical tensors (`h_input_dim + 1` embedding lookup) or continuous smooth distributions/logits (linearly projected via `smooth_h=True`).
*   **`t` ($\mathbb{R}^{N}$):** The continuous scalar diffusion timestep, localized per-atom.
*   **`edge_node_index` ($2 \times E$):** Bipartite indices mapping source and destination pairs, dictating which nodes interact directly as connected geometric bonds or $K$-nearest neighbor relationships.

---

## 2. Network Architecture Depth (`EquivEncoder`)

At the heart of the TRM_g3m framework lies the geometric graph neural network, explicitly engineered to process 3D molecular data while respecting the physical symmetries of the universe. The default instantiation is the `EquivEncoder` (`src_gmmm/nn/encoder.py`), representing an incredibly potent $SE(3)$-equivariant messaging network. 

### Dimensionality Definitions
Throughout the network, the variable **$F$** denotes the **`hidden_dim`** (configured in your `.yaml` settings, e.g., `hidden_dim: 256`).
*   **$F$** is the width of the invariant scalar track ($s$) and the feature dimension behind the equivariant vector track ($v \in \mathbb{R}^{3 \times F}$).
*   The value **$3F$** specifically appears in the Interaction Layer because the network predicts a "triple-header" embedding. This embedding is split into three separate vectors of size $F$ ($\phi_s, \phi_{vv}, \phi_{vs}$) to independently scale node-to-node scalar messages, node-to-node vector transfers, and bond-to-node structural transfers.

Before diving into the functional layers, it is critical to address how this architecture maps to the concepts outlined in the Equivariant Neural Diffusion (END) methodology:

> [!NOTE] 
> ### Connection to the Equivariant Neural Diffusion (END) Components
> The END paper establishes three distinct characteristics of equivariant diffusion models. Here is how the `g3m` framework natively implements (or omits) them:
>
> 1. **The Denoiser ($\theta$)**: **Present.** The `EquivEncoder` (wrapped inside `EquivariantParameterization` running in `DataPointReadout`) acts exactly as the "Data Point Predictor" $\hat{x}_\theta(z_t, t)$ (or as the noise predictor $\hat{\epsilon}_\theta$). It is called repeatedly during both the training loss calculations and the reverse inference trajectory to decode the noisy spatial configurations into their denoised ground-truth origins.
> 2. **Equivariance ($SE(3)$)**: **Present.** The explicit separation into invariant scalar tracks ($s$) and covariant vector tracks ($v$) throughout the architecture is precisely the mechanism that satisfies the strict $SE(3)$-equivariance requirements proven in Section 3.1 of the END paper. It guarantees that rotating a molecule before sending it into the network yields exactly the same outputs geometrically as rotating the network's outputs afterwards.
> 3. **The Learner ($\phi$)**: **NOT Present (by default).** The END paper prominently features a "Learnable Forward Process" where the encoder predicts a block-diagonal matrix $U_\phi(x, t)$ and mean $\mu_\phi(x, t)$ during corruption. The current `g3m` snapshot does *not* utilize this learnable forward parameterization. Instead, it relies on a fixed, non-learnable stochastic differential equation (`LinearLogSNRVPSDE`) to scale data and inject Gaussian noise, behaving more akin to a standard Equivariant Diffusion Model (EDM) than the full END formulation. To implement the Learner $\phi$, significant alterations to `src_gmmm/model/continuous.py` would be required.

### 2.1 The Two-Track Representation
The network's primary strategy for maintaining exact $SE(3)$ covariance (translations and rotations) is by bifurcating the data flowing through the nodes into two distinct representations per atom:
*   **Scalar State ($s$ $\in \mathbb{R}^{N \times F}$):** These tracks are strictly invariant. They contain scalar embeddings of categorical atomic species (e.g., carbon, oxygen) concatenated with scalar diffusion time embeddings. Any rotation of the input system leaves these variables perfectly unchanged.
*   **Vector State ($v$ $\in \mathbb{R}^{N \times F \times 3}$):** These tracks are equivariant directly mapped to 3D Cartesian coordinates. They are initialized as pure zeros to prevent any bias, maintaining physical directionality. When the atomic coordinates rotate, the internal values within this vector state rotate symmetrically.

### 2.2 Input Conditioning & Edge Expansions
Before the message-passing iteration begins, the model systematically unpacks geometric coordinates into continuous features.

1. **Diffusion Timestep Embedding (`FourierEmbedding`):** The continuous scalar time $t$ is projected into high-dimensional frequency space. Utilizing a `FourierEmbedding`, the network translates $1 \to 16$ sine/cosine frequencies using a learnable standard deviation projection matrix. It forces the network to inherently sense the timescale logarithmically. This time embedding is horizontally concatenated with the node element embeddings and cast into the $s$ track.
2. **Radial Basis Edges (`EdgeEmbedding`):** Geometric relationships must be encoded strictly relative to the central atom to guarantee translation invariance. 
   - **Normalized Direction:** The network extracts 1D structural directional vectors for edges ($x_{dest} - x_{source}$).
   - **Distance Extraction:** The scalar separations ($||x_{dest} - x_{source}||$) are captured.
   - **RBF Expansion:** These mere distances are mathematically expanded using exactly $64$ Gaussian offsets (Radial Basis Functions). This effectively acts as a continuous topological histogram, spreading the one-dimensional distance out into a 64-dimensional feature vector. 
   - **Cutoff Filter:** An explicit smooth spatial cutoff (`cosine_cutoff`) guarantees that distant interaction gradients cleanly decay to exactly $0.0$ at a configured cutoff radius (e.g., beyond $12.0 - 25.0 \mathring{A}$).

### 2.3 Layer Blocks: The Message Passing Protocol
The framework iterates through `num_layers` sequential blocks (e.g., 4 to 8 blocks). Each sequence pairs a message-passing `InteractionLayer` with a node-wise nonlinear `UpdateLayer`.

#### 2.3.a The Normalization Guard (`EquivLayerNorm`)
Standard `LayerNorm` would destructively interfere with the 3D coordinate frame. Therefore, the network restricts normalization utilizing a specialized mathematical approach. Prior to every interaction, scalar and vector tracks are normalized using graph-wide statistics. Given a graph $G$ with nodes $i$ and features $f \in \{1 \dots F\}$:

**Scalar Track ($s$):**
1. **Graph Mean:** $\mu_G = \text{mean}_{i \in G} (\text{mean}_{f} (s_{i,f}))$
2. **Graph Variance:** $\sigma^2_G = \text{mean}_{i \in G} (\text{mean}_{f} (s_{i,f} - \mu_G)^2)$
3. **Output:** $s'_{i,f} = \gamma_f \frac{s_{i,f} - \mu_G}{\sigma^2_G} + \beta_f$

**Vector Track ($v$):**
1. **Global Norm Scale:** $L^2_G = \text{mean}_{i \in G} (\text{mean}_{f} \|v_{i,f}\|^2)$
2. **Output:** $v'_{i,f} = \frac{v_{i,f}}{L^2_G}$

Mathematically, this ensures $s$ is centered and scaled by its spread, while $v$ is only scaled by its typical magnitude—preserving the absolute orientation of the vectors relative to the molecular center.

#### 2.3.b Edge-Centric Communication (`InteractionLayer`)
This function dictates exactly how nodes influence their neighbors geometrically by translating edge information and source scalar signals into updates. Given an edge $i \to j$ with unit vector $\vec{u}_{ij}$ and edge features $e_{ij}$:

1. **Combined Feature Context**:
   - Compute edge filter: $W_{ij} = \text{Linear}_{65 \to 3F}(e_{ij})$
   - Compute source signal: $\Phi_i = \text{MLP}_{F \to 3F}(s_i)$
   - Compute mixed message: $M_{ij} = W_{ij} \odot \Phi_i$ (Hadamard product)
   - Split $M_{ij}$ into three components: $\phi_{s, ij}, \phi_{vv, ij}, \phi_{vs, ij} \in \mathbb{R}^F$

2. **Edge Gating**:
   - $g_{ij} = \sigma(\text{Linear}_{F \to 1}(\phi_{s, ij}))$ (where $\sigma$ is the sigmoid function)

3. **Message Construction**:
   - **Scalar Message**: $m_{s, ij} = \phi_{s, ij} \cdot g_{ij}$
   - **Vector Message**: $m_{\vec{v}, ij} = \left( \vec{v}_{i} \odot \phi_{vv, ij} + \vec{u}_{ij} \otimes \phi_{vs, ij} \right) \cdot g_{ij}$
     - *Note*: $\vec{v}_i \odot \phi_{vv}$ scales the source vector's orientation, while $\vec{u}_{ij} \otimes \phi_{vs}$ promotes the inter-atomic unit vector into the vector feature space.

4. **Aggregation & Residual Update**:
   - $s_j^{(new)} = s_j + \sum_{i \in \mathcal{N}(j)} m_{s, ij}$
   - $\vec{v}_j^{(new)} = \vec{v}_j + \sum_{i \in \mathcal{N}(j)} m_{\vec{v}, ij}$

#### 2.3.c Node-Centric Synthesis (`UpdateLayer`)
Once a node has accumulated vectors and scalars from its neighbors, the `UpdateLayer` forces non-linear cross-communication between its own internal tracks:

1. **Vector Splitting & Norm Extraction**:
   - Split vector track: $(U_{v,i}, V_{v,i}) = \text{Linear}_{F \to 2F}(v_i)$ where $U_v, V_v \in \mathbb{R}^{3 \times F}$.
   - Extract invariant norms: $L_{V, i, f} = \|V_{v, i, f}\|_2$ (This "flattens" direction into rotation-invariant scalars).

2. **Coefficient Synthesis**:
   - $(a_{vv,i}, a_{sv,i}, a_{ss,i}) = \text{MLP}_{2F \to 3F}(\text{concat}(L_{V,i}, s_i))$

3. **Combined Update**:
   - **Scalar Update**: $\Delta s_i = a_{ss,i} + a_{sv,i} \odot (U_{v,i} \cdot V_{v,i})$ (Introduction of geometric angular alignment via spatial dot products).
   - **Vector Update**: $\Delta v_{i} = a_{vv,i} \odot U_{v,i}$ (Component-wise scaling of the $U_v$ track).

### 2.4 Structural Readout Extraction
The terminal `DataPointReadout` translates high-dimensional hidden states into physical outputs:

1. **Feature Track ($h$)**: $\hat{h}_i = \text{MLP}_{F \to \text{out\_dim}}(s_i)$.
2. **Positional Track ($pos$)**: 
   - Compress vector tracks: $\vec{v}_{readout, i} = \text{Linear}_{F \to 1}(v_i) \in \mathbb{R}^3$.
   - Apply residual shift: $\hat{x}_i = \vec{v}_{readout, i} + x_{i, \text{input}}$.
3. **Zero CoG**: $\hat{x}_G = \hat{x}_G - \text{mean}_{i \in G}(\hat{x}_i)$ (Strict removal of global translational drift).

---

## 3. Training Definition (Losses & Forward Process)

The overarching diffusion framework works uniformly across inputs over continuous time $t \in [0, 1]$, managed in `src_gmmm/model/continuous.py`.

### 3.1 The Forward Process Config
TRM_g3m leverages a **Linear Log-SNR Variance Preserving SDE (`LinearLogSNRVPSDE`)**.
The configuration explicitly bounds signal-to-noise ratio ranges with a `log_snr_min` and `log_snr_max` (e.g., `-10.0` to `10.0`). The log-SNR evolution function $\gamma(t)$ is linearly interpolated over $t$:
$$ \gamma(t) = \text{log\_snr\_min} + (\text{log\_snr\_max} - \text{log\_snr\_min}) \cdot t $$

The variance-preserving distributions dictating corruption profiles use strictly sigmoidal bounds parameterized from $\gamma(t)$. Specifically, the `loc` and `scale` are defined mathematically as:
*   **Scale of Clean Signal ($a_t$ / `loc`):** $\sqrt{\frac{1}{1 + \exp(\gamma(t))}} = \sqrt{\text{sigmoid}(-\gamma(t))}$
*   **Scale of Noise Signal ($b_t$ / `scale`):** $\sqrt{\frac{1}{1 + \exp(-\gamma(t))}} = \sqrt{\text{sigmoid}(\gamma(t))}$

**Forward Pass Generation:** 
$$ x_t = \text{loc}(t) \cdot x_0 + \text{scale}(t) \cdot \epsilon $$

### 3.2 Continuous Categorical Diffusion & Loss
A common point of confusion is how **MSE** can be applied to categorical variables like atom types ($h$). TRM_g3m utilizes a "Continuous Relaxation" strategy to treat discrete labels as spatial coordinates:

1. **One-Hot Mapping**: Discrete categorical labels are first transformed into one-hot vectors $\mathbf{h}_0 \in \{0, 1\}^K$. These are handled as continuous float tensors in Euclidean space.
2. **Diffusion in Embedding Space**: The forward SDE applies Gaussian noise $\epsilon \in \mathbb{R}^K$ to these vectors:
   $$\mathbf{h}_t = \text{loc}(t) \cdot \mathbf{h}_0 + \text{scale}(t) \cdot \epsilon$$
   This turns the discrete point (the one-hot corner of a hypercube) into a continuous Gaussian cloud.
3. **The Loss**: The model predicts a continuous-valued vector $\hat{\mathbf{h}}_0$. The training loss computes the Euclidean distance between the predicted feature vector and the clean one-hot target:
   $$\mathcal{L}_{h} = \text{MSE}(\hat{\mathbf{h}}_0, \mathbf{h}_0)$$
4. **Discretization**: During the reverse sampling process, the model generates continuous-valued feature vectors. These are mapped back to discrete atom types by taking the **argmax** across the $K$ dimensions of the final generated state.

The unified training loss (`EquivariantDiffusion.loss_diffusion` $\to$ `ContinuousDiffusion.loss_diffusion`) thus minimizes the joint error in coordinate space and feature-embedding space:
$$ \mathcal{L}_{\text{total}} = \text{MSE}(\text{Pred}_{\text{pos}}, \text{Target}_{\text{pos}}) + \text{MSE}(\text{Pred}_{h}, \text{Target}_{h}) $$

### 3.3 Prediction Targets ($x_0$ vs. $\epsilon$)
Whether the network learns to "denoise" (predict the clean state) or "un-noise" (predict the noise) is a global hyperparameter set via `parameterization` in the `.yaml` config (e.g., `parameterization: "x0"`).

#### Training Targets
Depending on this setting, the `training_targets` function in `src_gmmm/model/continuous.py` assigns the loss target $\mathbf{y}$:
*   **`parameterization: "x0"`**: The target is the original data point: $\mathbf{y} = x_0$.
*   **`parameterization: "eps"`**: The target is the Gaussian noise vector: $\mathbf{y} = \epsilon$.

#### Score Construction
Because the reverse SDE requires the **Score Function** $\nabla_x \log p_t(x)$, the model must transform its predicted value ($\hat{x}_0$ or $\hat{\epsilon}$) into a score during inference:

| Parameterization | Score Formula ($s_\theta$) |
| :--- | :--- |
| **`x0`** | $s_\theta = \frac{\text{loc}(t) \cdot \hat{x}_0 - x_t}{\text{scale}(t)^2}$ |
| **`eps`** | $s_\theta = - \frac{\hat{\epsilon}}{\text{scale}(t)}$ |

---

## 4. Inference / Reverse Sampling Definition
Inference is integrated downward across an explicit timeline traversing steps $i \in \{n\_steps, \dots, 0\}$.

### 4.1 Constructing the Score Tensor Form
Whether predicting noise ($\hat{\epsilon}$) or states directly ($\hat{x}_0$), outputs translate into standardized deterministic score maps natively structured to reverse the SDE process: $s_\theta(x_t, t) \approx \nabla_{x} \log p_t(x)$.
*   **Predicts "eps":**  $$ s_{\theta} = - \frac{\hat{\epsilon}}{\text{scale}(t)} $$
*   **Predicts "x0":**  $$ s_{\theta} = \frac{\text{loc}(t) \cdot \hat{x}_0 - x_t}{\text{scale}(t)^2} $$

### 4.2 Euler-Maruyama Mapping (`reverse_step_em`)
The discretization iteratively resolves reversing components across continuous $t$ stepping via $\Delta t$:
*   **Forward Linear Drift ($f$):**  $- 0.5 \cdot \beta(t) \cdot x_t$
*   **Volatility Output ($g$):** $\sqrt{\beta(t)}$, where $\beta(t)$ maps precisely as the time-derivative of the Log SNR limits.

The numerical iteration evaluating via Standard Euler updates:
$$ x_{t - \Delta t} = x_t + \left[ -0.5 \cdot \beta(t) \cdot x_t - 0.5 \cdot g(t)^2 \cdot s_{\theta} \right] \Delta t + g(t) \sqrt{\Delta t} \cdot \epsilon_{\text{standard}} $$

---

## 5. Configuration Settings & Architectures
TRM_g3m relies on **Hydra** combined with OmegaConf (`configs/*.yaml`) to parameterize runs using dynamic instantiation (`_target_`).

### 5.1 Native Architectures
Currently, the `src_gmmm` module natively provides one core geometric encoder:
*   **`_target_: src_gmmm.nn.encoder.EquivEncoder`:** The default equivariant network running on paired $SE(3)$ scalar and vector tracks.

### 5.2 Selecting Alternative Architectures via YAML
Due to Hydra's dynamic instantiation, you are not locked into the `EquivEncoder`. You can hot-swap the model architecture directly from the YAML file, provided the new class adheres to the same forward pass signature (taking `t, h, pos, node_index, edge_node_index` and returning `{s, v}`).

If you implement or import a different geometric neural network (like an EGNN), you can select it by changing the `.yaml`:

```yaml
      encoder:
        _target_: path.to.your_custom_architecture.CustomEGNNEncoder
        hidden_dim: ${hidden_dim}
        num_layers: ${num_layers}
```

This flexibility means you can rapidly test variations or entirely disparate message-passing formats purely via settings without rewriting the PyTorch Lightning diffusion loops.

---

## 6. Comprehensive File Overview (`src_gmmm`)

Below is a complete index of the active Python modules inside the `src_gmmm` package, categorized by their engineering roles. 

### Data Flow (`src_gmmm/data/`)
*   **`datamodule.py`**: A PyTorch Lightning DataModule that orchestrates dataset loading, subset splitting, and mapping graphs to dataloaders.
    *   *Key Classes*: `DataModule`
*   **`dataset.py`**: Defines custom `Dataset` structures to serialize raw chemical graphs and instantiate empirical starting distributions.
    *   *Key Classes*: `Dataset`, `SampleDataset`
*   **`transforms.py`**: Intercepts geometric graphs during loading to apply on-the-fly transformations before reaching the batcher.
    *   *Key Classes*: `OneHot` (discrete atom types to continuous float vectors), `FullyConnected` (bond matrices), `ZeroCoG` (origin centering).
*   **`utils.py`**: Provides auxiliary parsing functions to handle dataset metadata and decode tensors back into human-readable chemical forms.
    *   *Key Functions*: `atoms_from_tensors`, `read_json`

### Training Lifecycle (`src_gmmm/lit/`)
*   **`module.py`**: The definitive PyTorch Lightning system encapsulating training steps, validation loops, loss evaluation, and Exponential Moving Average (EMA) weight tracking.
    *   *Key Classes/Functions*: `LitModule`, `basic_step()` (calculating SDE loss), `sample()` (running Euler-Maruyama diffusion inference).

### Metrics & Evaluation (`src_gmmm/metrics/`)
*   **`base.py`**: Holds the abstract parent templates and fundamental math ops for any metric pipeline reporting generator quality.
    *   *Key Classes/Functions*: `Metrics`, `discrete_histogram`
*   **`bonds.py`**: Executes strict computational chemistry checks to tally assigned physical valencies against known atomic stability limits.
    *   *Key Functions*: `check_stability`
*   **`qm9.py`**: Accumulates the three canonical generational benchmarks (validity, uniqueness, novelty) specifically tailored for the QM9 dataset bounds.
    *   *Key Classes*: `QM9Metrics`
*   **`rdkit_utils.py`**: Forms a bridge to the standard RDKit library to sanitize generated 3D clouds into valid canonical SMILES strings.
    *   *Key Functions*: `make_mol_rdkit_qm9`, `check_validity`

### Core Model & SDE Logic (`src_gmmm/model/`)
*   **`continuous.py`**: Calculates the strict mathematical forward-process boundaries (the stochastic differential equations).
    *   *Key Classes*: `SDE`, `LinearLogSNRVPSDE` (the exact SNR bounding logic), `ContinuousDiffusion`
*   **`diffusion.py`**: The top-level architect combining continuous diffusion tracks independently for positions and categories.
    *   *Key Classes/Functions*: `EquivariantDiffusion`, `loss_diffusion()`, `sample()`, `reverse_step_em()`
*   **`distributions.py`**: Defines the foundational continuous multivariate normal priors spanning the 3D generation bounds.
    *   *Key Classes*: `DistributionGaussian`
*   **`score.py`**: A wrapper to package arbitrary geometric encoders and readouts underneath a unified predictor interface intended for SDE scoring.
    *   *Key Classes*: `EquivariantParameterization`

### Neural Network Architecture (`src_gmmm/nn/`)
*   **`encoder.py`**: Houses the heavy-duty geometric graph layers responsible for covariant message passing and nonlinear tracking.
    *   *Key Classes*: `InteractionLayer`, `UpdateLayer`, `EquivEncoder`
*   **`layers.py`**: Contains specialized embeddings and mathematically constrained normalization modules to protect 3D symmetry.
    *   *Key Classes*: `FourierEmbedding` (timestep projection), `EdgeEmbedding` (RBF creation), `EquivLayerNorm`
*   **`readout.py`**: Interprets the deep embeddings from the final layer back into raw spatial coordinates and atomic logits.
    *   *Key Classes*: `DataPointReadout`

### Utilitarian (`src_gmmm/utils/`)
*   **`eval.py`**: Main script explicitly built to process validation runs natively on loaded checkpoints without triggering backward passes.
*   **`callback.py`**: Holds Pytorch Lightning hooks that execute intermittently to survey graph generation progress.
    *   *Key Classes*: `LogSampledAtomsCallback`
*   **`ops.py`**: Dedicated low-level PyTorch tensor mathematical operations.
    *   *Key Functions*: `scatter_center`, `center`
*   **`pylogger.py` / `rich_utils.py`**: Provides visual terminal aesthetics and clean logging formatting for reading active configs.
*   **`utils.py`**: Orchestrates high-level system functions like binding loggers, flushing metrics to W&B, and saving hyperparameters safely.

### Entry (`src_gmmm/`)
*   **`train.py`**: The authoritative CLI script wrapped in `hydra` to orchestrate settings, build the trainer, and trigger the training loop.