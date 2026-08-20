# Generative modeling for Molecules and Materials

## Installation

```
# clone this repo
git clone https://github.com/frcnt/g3m.git

# move to the root directory
cd gm3m/

# create an environment
pip install uv
uv venv .venv --python 3.11
source .venv/bin/activate


uv pip install -e .
```

## Getting Started

### 1. Pre-processing the GEOM-Drugs Dataset

To preprocess the `GEOM-Drugs` dataset with the provided splits, run the following command. 
It downloads the dataset, filters molecules by atom count, generates pyG `Data` objects, and extracts necessary MACCS keys and coordinate information.

*(Note: All required dataset generation scripts are already included in the `scripts/` directory. No additional untracked scripts are needed.)*

```bash
export TARGET_DIR="data/geom_drugs"
export SPLIT_FILE="data/geom_drugs/splits.json"

python scripts/preprocess_geomdrugs.py --target_dir $TARGET_DIR --split_file $SPLIT_FILE
```

### 2. Computing Eigenvalues (Conditioning Features)

The model utilizes eigenvalues of the coordinate covariance matrix as shape conditioning features ($c$). 
To compute and append these to your preprocessed `.pt` dataset files, run:

```bash
python scripts/compute_eigenvalues.py --data_dir data/geom_drugs
```

**How Eigenvalues are Computed:**
1. **Centering**: For each molecule, the 3D coordinates ($X$) are mean-centered ($X_c$).
2. **Covariance**: The unnormalized covariance matrix is computed as $X_c^T X_c$.
3. **Eigenvalues**: The 3 eigenvalues ($\lambda_1 \ge \lambda_2 \ge \lambda_3 \ge 0$) of this matrix are extracted.
4. **Linearization & Normalization**: The eigenvalues are linearized via square root $E_i = \sqrt{\lambda_i}$. A global scale $S = \sum E_i$ is computed, and the fractional shape normalizations $\hat{s}_i = E_i / S$ are derived.
5. **Conditioning Vector**: The final conditioning vector is the concatenation $c = [\hat{s}_1, \hat{s}_2, \hat{s}_3, S]$.

### 3. Launching a Training Run

By default, the example config expects the environment variables `DATA_PATH` and `LOG_PATH` to be defined. Below is a concise example of how to launch a training job, similar to a cluster jobscript, with explanations for key architectural arguments.

```bash
export DATA_PATH="data" # Path containing the preprocessed dataset
export LOG_PATH="output"
export CONFIG_NAME="train_geom_sp"

python src_gmmm/train.py -cn $CONFIG_NAME \
    n_integration_steps=250 \
    lit_module.lr=2e-4 \
    self_conditioning=false \
    latent_recursion_training_method=standard \
    latent_recursion="edge_mlp_update" \
    num_layers=4 \
    n=1 \
    scprop=0.5 \
    K=3 \
    ablations.ablate_s=false \
    ablations.ablate_v=false \
    ablations.ablate_edge=false \
    ablations.ablate_unit_vectors=true
```

**Key Training Arguments:**
* `self_conditioning` (bool): Enables standard data-driven self-conditioning. **⚠️ WARNING: This is mutually exclusive with Latent Recursion.** If using `latent_recursion`, set this to `false`.
* `latent_recursion_training_method` (str): Sets the unrolling strategy during training (e.g., `standard` for normal BPTT, `stop_grad`).
* `latent_recursion` (str): Defines the architectural mechanism used for recurrent updates. For example, `"edge_mlp_update"` updates edges using geometric features from the vector states while applying skip-connections to node states.
* `num_layers` (int): The number of Message Passing (MP) layers per timestep. Depth scales inference time linearly but adds a large fixed overhead.
* `n` (int): The number of unrolled latent recursion steps to execute during training forward passes.
* `K` (int): The number of independent latent recursion samples/chains to maintain.
* `scprop` (float): The probability of applying the latent recursion conditioning (similar to dropout for guidance).
* `ablations.*`: Booleans to selectively ablate (disable) specific features like scalar representations (`ablate_s`), vector representations (`ablate_v`), or unit vectors (`ablate_unit_vectors`).

## Inference & Denoising

The diffusion model denoises 3D coordinates and atom types over a continuous time schedule. Below is a 250-step denoising trajectory of a test-set molecule using a trained self-conditioned RBF model snapshot, along with its ground truth (true) structure for comparison:

| Denoising Animation | True Molecule |
| :---: | :---: |
| ![Denoising Animation](evals/denoising_animation.gif) | ![True Molecule](evals/true_molecule.png) |

*(Generated via `scripts/animate_denoising.py`)*
