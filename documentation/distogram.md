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

    