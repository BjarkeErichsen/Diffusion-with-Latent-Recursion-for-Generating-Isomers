Adding the distogram similar to alphafold and other models.


Steps:

    0 Create a script that adds the pairwise distances histogram feature to every datapoint in the GEOM drugs dataset.
        - It should add the feature similarly to how eigenvaleus was added to the dataset
        - It should be normalized to a probability distribution Ie. its a pure probability distribution, thats storted for each molecule. 
        - In this we select Number of binds = 128 (same as alphafold)
        - Cutoff for distances is 3 Angstroms.
        - The histogram represents the pairwise distances of every atom to every other atom
        - Additionally, add an ipunb file called explore_distograms, this file assumes we have already run the preprocessing scirpt that adds distograms. 


    1 Addition of a distogram feature to the config file.
        - use_dist boool
        - distogram_loss_coefficient float
        (the values of cutoff distance and number of buckets is set when i create the dataset, so have these be variables that are set as default inside the code base not in the config file. )
       
Loss:
    Computed using multiclass cross entropy.

Implementation details:
    1 Which representation(s) should we predict the histogram from?
        - the edge representation. 
    2 How should the histogram head be constructed?
        answer = a single linear layer with bias (same as alphafold)
    3 Coefficient of loss on distogram vs other loss?
        distogram_loss_coefficient = 0.1, lets start there. 
    4 cutoff:
        Should be around 3.2 as that is the exclusive cutoff distance for van der waals bonds.
        We start at 0, as we want to make sure no non-bonded atoms are predicted to be too close to each other.
    5 To the metrics add a new plot thats also pushed to wandb called distogram loss, this should be done for both validation and training. Ie we plot both the full training loss as we already do and aadditionally the specific components of it thats the distogram loss. 
    6 add the distogram head function to src_gmmm/nn/readout.py. Please keep the code as clean as possible, dotn add unneccessary pritn statements, comments or otherwise. Dont fuzz to much on security / type hints, focus on just getting the job done.
    7 Write to this file a description of what youive implemented, sepcficially referencing the code youve added. 

### Implementation Description:
1. **Preprocessing Script & Notebook (Deprecated)**:
   - *Previous precomputation lines deleted*: We no longer precompute or explore distograms in the datasets. The discretization and distance calculations are performed on the fly during training.
2. **Architecture (`src_gmmm/nn/readout.py`)**:
   - Modified `DataPointReadout` to conditionally accept `pred_distogram=True`.
   - The predictive head is a single linear layer `self.net_dist = nn.Linear(hidden_dim, 128)` which outputs 128 logits per edge.
   - *Deleted* the `scatter_mean` node-pooling logic. The readout now correctly outputs `out["dist"]` directly with shape `[num_edges, 128]` to retain edge-wise predictions.
3. **Loss Computation (`src_gmmm/model/diffusion.py`)**:
   - *Deleted* the old logic that fetched a precomputed `batch.distogram` probability distribution.
   - *Added* a dedicated, isolated `DistogramLoss` `nn.Module` class to handle the scaling constraints cleanly.
   - **`DistogramLoss` Mechanics**:
     - Pulls `u, v` from `batch.edge_node_index` and computes the exact true distances `dists = torch.norm(pos[u] - pos[v], dim=-1)` for all provided edges.
     - Assuming the graph is constructed strictly as an upper-triangle (without symmetric duplicates or self-loops), the loss naturally computes the correct average. If this assumption is violated (`u >= v` is found), it prints a warning instead of forcibly filtering.
     - Bucketizes the distances into 128 discrete bins. Distances > 3.2Å are clamped into the final category (`target_bins = torch.clamp(..., max=127)`).
     - Computes `torch.nn.functional.cross_entropy` over the edges and averages the loss (`reduction="mean"`), scaling efficiently without dense memory instantiation.
   - In `EquivariantDiffusion.loss_diffusion`, the `DistogramLoss` is evaluated on-the-fly passing the true positions (`batch.pos`), the fully connected graph edges (`batch.edge_node_index`), and the edge predictions (`preds["dist"]`).




Updated version: (previous version was incorrect)
    1. (dont do this for now).
        Dont pre-compute the pairwise distances / discritization. Compute them during the loss computation step.

    2. For each edge we should predict a distogram, this distogram pertains to the exact distance of the two atoms connected by that edge. 
    3. We utilize the same loss function but we need to sum the losses for every edge.
    4. Problem:  This will cost too much computation, due to the number of edges being too large.... (we ignore this problem for now...)
    5. Consider every distance greater than the cutoff as its own category. 
    6. We still use a single layer (no hidden layer) predictive head for the edge (same for each edge of course. )
        - Edges are not symmetric, ie. we only write a->b not b->a, this is fine but it means we only get an upper triangle (with no diagonal) part of the matrix.
    7. Store all the distances computed and real in a matrix. its of shape (num_atoms, num_atoms, num_bins)
        Compute a combined loss for all.
        Average the loss across num_atoms * num_atoms / 2 - num_atoms
        (remember it ends up being an upper triangle matrix without a diagonal)
    
    8. Isolate code when possible in functions and classes to do this (especially classes)
    9. Continue the previus implementation description that describes where and what you change, delete the previous lines, when you override/delete previous parts of the implementaiton of distogram loss. 
    10. of course, this is not done with hydrogens added, we are doing all of this in the no_hydrogen scenario. 


Discussion solutions to the scaling issue: (do not consider for now)
    Trick: We only consider those edges that are predicted to be close in the space of positions?

    Trick: Consider only edges close in the target molecule (the true position space).