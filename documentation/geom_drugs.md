1. Standard Download MethodThe most computationally efficient way to retrieve GEOM-Drugs is via the official MIT geom GitHub repository, specifically downloading the drugs_summary.json and the rdkit_folder.tar.gz files.Parsing the monolithic 37-million-conformer .msgpack file directly from scratch is highly inefficient and requires massive RAM. The standard pipeline utilizes the lightweight JSON summary file to pre-filter your dataset (e.g., establishing your subset) before extracting the heavier 3D RDKit objects from the compressed tarball.  2. Preparation PipelineBy reading the summary JSON, you can select your subset to limit compute spend and then extract only those specific .pickle files from the archive. For the functional group hashing, MACCS keys will evaluate the structure and return a fixed-size binary vector $h \in \{0, 1\}^{167}$.3. Best Visualization Method: mols2gridWhile RDKit's PandasTools is functional, it embeds static images directly into the DataFrame, which severely bloats memory and can break standard Pandas operations. The modern standard for this is mols2grid. It reads the SMILES strings from your Pandas DataFrame and renders an interactive, paginated, and searchable HTML grid directly in a Jupyter cell, leaving the underlying DataFrame lightweight.ImplementationYou will need to install the visualizer: pip install mols2grid


What i want to implement:
    n atom cutoff (default 40) 
    
Implement this in a script similar to
scripts/preprocess_qm9.py
      Take HEAVY inspiration from this except when it comes to the practicals of getting the geomdataset in the first place. 
Dataset should be donwloaded similar to how its done in scripts/preprocess_qm9.py, to a specific folder with the appropriate name and partitions into train, val test. 

12. Implement a .pynb notebook showing how to use the dataset after preprocessing for exploration. Should perform HIGHLY similarly to scripts/explore_data.ipynb does for qm9 but adapted for geomdrugs. 

## Added Files and Utility

1. **`src_gmmm/metrics/geom_drugs.py`**:
   - Implements `GeomDrugsMetrics` to evaluate the chemical validity and 3D stability of the 15 extended elements present in GEOM-Drugs.
2. **`scripts/preprocess_geomdrugs.py`**:
   - Downloads the dataset tarball and filters molecules by atom count <= 40.
   - Extracts and processes the matching conformers sequentially to construct train/val/test splits saved as PyG `.pt` and `_infos.json` metadata files.
3. **`scripts/explore_geomdrugs.ipynb`**:
   - Jupyter Notebook showing how to load the preprocessed splits, inspect PyG object properties, plot centered 3D conformer coordinates, and render interactive searchable molecular tables using `mols2grid`.