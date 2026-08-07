# Agent Rules

- For all future evaluation runs (evals), ALWAYS use the `gpuv100` queue with a time limit of 3 hours (`-W 03:00`). NEVER use the `a100` (`gpua100`) queue for these evaluation runs, regardless of what cluster the original ablation/training experiment was run on.

- **Folder Structure & Checkpoints**: 
  - Model weights/checkpoints are stored in paths similar to: `snapshots/<category>/<experiment_name>/output/geomdrugs/runs/<timestamp>/checkpoints/`.
  - **IMPORTANT**: For ablation experiments (especially those run on `v100`), there are often two separate timestamp folders in `output/geomdrugs/runs/`. **ALWAYS use the chronologically later folder**, as the first one will not contain the best epoch 99.9% of the time.

- **Experiment Continuation Protocol**: 
  When resuming/continuing an unconverged experiment, you must follow these steps to avoid configuration mismatches or WandB trajectory breaks:
  - ALWAYS explicitly `cd` into the experiment's snapshot directory to ensure you are using its exact code and config.
  - Set absolute paths for `$DATA_PATH`, `$PYTHONPATH`, and `$LOG_PATH`.
  - Specify the exact latest checkpoint using `+ckpt_path=$PWD/output/geomdrugs/runs/<latest_timestamp>/checkpoints/last.ckpt`.
  - Seamlessly continue the WandB trajectory by extracting the original run ID from the wandb folder and passing: `logger.wandb.id=<original_wandb_id> +logger.wandb.resume="must"`.
  - Pass the exact same ablation parameters (e.g., `ablations.ablate_edge=true`) originally used, which can be verified in the run's `.hydra/overrides.yaml`.

- **Evaluating Snapshots & Legacy Checkpoints**:
  - When evaluating an older checkpoint (e.g., via `eval_best_model.py`), **always** ensure the script loads the exact code from the snapshot directory rather than the actively modified codebase in the workspace root.
  - Achieve this by modifying the bash submission scripts (e.g., setting `PYTHONPATH` correctly to prioritize the snapshot's code directory) rather than altering the shared evaluation python scripts (e.g., do NOT hardcode absolute paths in `eval_best_model.py`).

- **File Output Locations**:
  - NEVER save plots, outputs, or any files outside of the user's Current Working Directory (CWD) or explicitly specified project directories unless explicitly asked to do so. NEVER save to the internal IDE scratch directory.

- **Primary Dataset Configuration**:
  - ALWAYS inspect the GEOM-Drugs configuration file (`configs/train_geom_sp.yaml`) first when answering questions about how model components, hyperparameters, or training/evaluation workflows work in this repository.

