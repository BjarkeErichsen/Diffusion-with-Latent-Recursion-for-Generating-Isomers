# Classifier free guidance on GEOM or QM9

## Setup on molecule generation models

Compute the associated eigenvaleu for each molecule in the dataset.

Append this feature to the dataset.

Todo:
    1 Make a script inside scripts, that does this!
    2 add a demonstration in an inpynb that shows computing the eigenvalues for a subset of molecules, visualizing the computed values with the molecules.
    3 Remember, this needs to be relatively fast as we have many datapoitns

## Which attribute to use
Shape conditioning: 
    Condition on eigenvalues of the atom positions.  

Note:
    * whether to use hydrogen atoms or should be experimented with. For now Do use hydrogens.



## Evals for GEOM drugs [Implemented]

The RDKit-based evaluation metrics for GEOM drugs have been directly updated in `src_gmmm/metrics/geom_drugs.py` to support hydrogen-free evaluations:
1. **Hydrogen Filtering**: All hydrogen ("H") atoms are excluded before reconstructing the molecular graph.
2. **Object Creation**: An RDKit molecule object is initialized with heavy atoms and their generated 3D coordinates.
3. **Bonds & Valency assignment**: Connectivity and bond orders are determined using `rdDetermineBonds.DetermineBonds` with a multi-charge loop fallback (`[0, 1, -1, 2, -2]`).
4. **Hydrogen Creation**: Implicit hydrogens are converted to explicit 3D atoms using `Chem.AddHs(mol, addCoords=True)`.
5. **Validity & Stability Definition**:
   - `valid`: Sanitization succeeds on the molecule with added hydrogens (`mol_with_hs`).
   - `molecule_stable`: The heavy-atom molecule (`mol`) is valid without adding any implicit hydrogens (`total_implicit_hs == 0`).
   - `valency_table_molecule_stable` & `valency_table_atom_stable`: Checked against the aromatic-aware tuple valency table from `isayevlab/geom-drugs-3dgen-evaluation`.
   - `atom_stable`: Counts heavy atoms with `0` implicit hydrogens.



    
## CFG variant

For starters we use standard guidance as we would do with any conditional property.

$\tilde{\epsilon}_t = \epsilon_\theta(x_t, \emptyset) + w \cdot (\epsilon_\theta(x_t, c) - \epsilon_\theta(x_t, \emptyset))$

Use standard 50/50% trainign with and without conditioning. 

(skip for now) However given that the shape property is 1) computable and 2) differentiable we can use more informative measures:
1. Inference time optimization (universal guidance)
2. (better) Train-time optimization (adding a term to the loss function during training)
We compute the value during training and calculate a loss from this. 
Shape is invariant to rotation. The diagonal values are ordered so its effectively permutation invariant and the loss thus properly defined. 



## Implementation in the GNN 


We dont do cross attention!
We implement it in a normalization parameter inside EquivLayerNorm 
    Its already implemented there.

To do:
    1 Make sure the conditional EquivLayerNorm works
        1. Ive only changed the part inside the self.condition_dim is not none parts
        2. The v part should keep equivariance property!
    2 Interate it into the rest of the code such that when a conditional property as part of the input, we use it!

Why?
    For relatively low amounts of information like shape or other scalar values, this is more appropriate.
        As contrasted with cross-attention which is for high-information conditional information. 



Note:
    read documentation/architecture_context.md

    Please add to configs/train_geom_sp.yaml the following:
        cfg: true/false
        cfg_prop: 0.5
        cfg_property: "eigenvalues" or "eigenvalues_normalized"




## Evals
Please as best you can isolate evals to do with CFG in its own seperate file. 

### 1
How well does the conditional property during validation 