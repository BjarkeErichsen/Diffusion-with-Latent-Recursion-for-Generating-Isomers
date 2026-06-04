import os.path
import tarfile
import urllib.request
import pickle
import json
from pathlib import Path
from typing import Optional

import ase
import fire
import numpy as np
import torch
import tqdm
from torch_geometric.data import Data

from src_gmmm import utils
from src_gmmm.data.utils import read_json, save_json
from src_gmmm.metrics.geom_drugs import GeomDrugsMetrics
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

log = utils.get_pylogger(__name__)

SEED = 0
URL_GEOMDRUGS = "https://dataverse.harvard.edu/api/access/datafile/4327252"


def count_atoms_including_h(smiles: str) -> Optional[int]:
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mol = Chem.AddHs(mol)
        return mol.GetNumAtoms()
    except Exception:
        return None


def download_url(url: str, dest_path: str):
    log.info(f"Downloading '{url}' to '{dest_path}'...")
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36"
        }
    )
    with urllib.request.urlopen(req) as response:
        total_size = int(response.headers.get("content-length", 0))
        block_size = 1024 * 1024  # 1 MB blocks
        
        with open(dest_path, "wb") as f:
            if total_size > 0:
                with tqdm.tqdm(total=total_size, unit="iB", unit_scale=True, desc="Downloading") as pbar:
                    while True:
                        buffer = response.read(block_size)
                        if not buffer:
                            break
                        f.write(buffer)
                        pbar.update(len(buffer))
            else:
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    f.write(buffer)


def preprocess_geomdrugs(
    target_dir: str | Path = "data/geom_drugs/",
    split_file: Optional[str | Path] = None,
    n_atom_cutoff: int = 40,
):
    # Setup directories
    download_dir = os.path.join(target_dir, "download")
    os.makedirs(download_dir, exist_ok=True)
    log.info(f"The downloaded files will be placed in '{download_dir}'.")

    fname_tar = os.path.join(download_dir, "rdkit_folder.tar.gz")
    if not os.path.exists(fname_tar):
        download_url(URL_GEOMDRUGS, fname_tar)
        log.info("Done downloading.")

    log.info("Opening tarball...")
    with tarfile.open(fname_tar, "r") as tf:
        log.info("Locating summary JSON inside tarball...")
        
        # Try known candidate paths to extract the summary JSON without scanning
        summary_file = None
        for candidate in [
            "rdkit_folder/summary_drugs.json",
            "rdkit_folder/drugs_summary.json",
            "summary_drugs.json",
            "drugs_summary.json"
        ]:
            try:
                summary_file = tf.extractfile(candidate)
                log.info(f"Found summary JSON directly at: {candidate}")
                break
            except KeyError:
                continue

        if summary_file is None:
            log.info("Direct extraction failed. Scanning tarball sequentially for summary JSON...")
            for member in tf:
                if member.name.endswith("summary_drugs.json") or member.name.endswith("drugs_summary.json"):
                    summary_file = tf.extractfile(member)
                    log.info(f"Found summary JSON sequentially at: {member.name}")
                    break

        if summary_file is None:
            raise FileNotFoundError("Could not find summary_drugs.json inside the tarball.")
        
        drugs_summ = json.load(summary_file)

        log.info("Filtering molecules by atom count <= {} (including explicit hydrogens)...".format(n_atom_cutoff))
        filtered_smiles = []
        for smiles in tqdm.tqdm(drugs_summ.keys(), desc="Filtering molecules", unit="mol", mininterval=5.0):
            n_atoms = count_atoms_including_h(smiles)
            if n_atoms is not None and n_atoms <= n_atom_cutoff:
                filtered_smiles.append(smiles)
        
        log.info(f"Found {len(filtered_smiles)} molecules matching the cutoff out of {len(drugs_summ)} total molecules.")

        # Split definition
        split_path = os.path.join(target_dir, "splits.json")
        if split_file:
            log.info(f"Loading the provided split file: '{split_file}'.")
            splits = read_json(json_path=split_file)
            splits = {split: list(sorted(splits[split])) for split in splits}
        else:
            log.info("Creating the splits.")
            n_indices = len(filtered_smiles)
            n_train = min(100000, int(0.8 * n_indices))
            n_test = int(0.1 * n_indices)
            n_val = n_indices - (n_train + n_test)

            np.random.seed(SEED)
            data_perm = np.random.permutation(n_indices)
            train_ptr, val_ptr, test_ptr = np.split(
                data_perm, [n_train, n_train + n_val]
            )

            splits = {
                "train": [filtered_smiles[i] for i in sorted(train_ptr.tolist())],
                "val": [filtered_smiles[i] for i in sorted(val_ptr.tolist())],
                "test": [filtered_smiles[i] for i in sorted(test_ptr.tolist())],
            }
            log.info("Done creating the splits.")
            log.info(f"Saving the splits for later reuse to: '{split_path}'.")
            save_json(json_dict=splits, json_path=split_path)

        # Build mapping of required pickle files for fast lookup during sequential scan
        log.info("Building mapping of required pickle files...")
        required_paths = {}
        for split in splits:
            for smiles in splits[split]:
                pickle_path = drugs_summ[smiles].get("pickle_path")
                if pickle_path:
                    # Map both the raw path and alternative versions (with/without prefix)
                    required_paths[pickle_path] = (split, smiles)
                    if pickle_path.startswith("rdkit_folder/"):
                        required_paths[pickle_path[len("rdkit_folder/"):]] = (split, smiles)
                    else:
                        required_paths[os.path.join("rdkit_folder", pickle_path)] = (split, smiles)

        preprocessed_dir = os.path.join(target_dir, "preprocessed")
        os.makedirs(preprocessed_dir, exist_ok=True)

        from rdkit.Chem import MACCSkeys

        # Initialize split lists and metrics
        split_metrics = {
            split: GeomDrugsMetrics(max_num_atoms=n_atom_cutoff, summarize_hidden=True, hidden_prefix="")
            for split in splits
        }
        split_lists = {split: [] for split in splits}

        log.info("Extracting and processing conformers sequentially from tarball...")
        pbar = tqdm.tqdm(total=len(required_paths) // 3, desc="Processing conformers", unit="mol", mininterval=10.0)

        # Iterate over tarball sequentially to decompress only once and avoid seeking
        for member in tf:
            if not member.isfile():
                continue
            if member.name not in required_paths:
                continue
            
            split, smiles = required_paths[member.name]
            
            try:
                file = tf.extractfile(member)
                data_dict = pickle.load(file)
            except Exception as e:
                log.warning(f"Error loading pickle for {smiles}: {e}")
                continue

            if not data_dict.get('conformers'):
                continue
            conformer = data_dict['conformers'][0]
            rd_mol = conformer.get('rd_mol')
            if rd_mol is None:
                continue

            # Get conformer positions and center them
            conf = rd_mol.GetConformer()
            positions = conf.GetPositions()
            positions = positions - positions.mean(axis=0)

            # Build ASE Atoms for validation metric tracking
            symbols = [atom.GetSymbol() for atom in rd_mol.GetAtoms()]
            atoms = ase.Atoms(symbols=symbols, positions=positions)
            split_metrics[split]([atoms])

            # Get MACCS keys
            maccs = MACCSkeys.GenMACCSKeys(rd_mol)
            maccs_list = [int(bit) for bit in maccs]

            # PyG Data Object
            data = Data(
                h=torch.LongTensor([atom.GetAtomicNum() for atom in rd_mol.GetAtoms()]),
                pos=torch.Tensor(positions),
                maccs=torch.LongTensor(maccs_list),
                smiles=smiles
            )
            split_lists[split].append(data)
            pbar.update(1)

        pbar.close()

        for split in splits:
            log.info(f"Summarizing split information for '{split}'...")
            infos_path = os.path.join(preprocessed_dir, f"{split}_infos.json")
            agg_infos = split_metrics[split].summarize()
            save_json(agg_infos, infos_path)

            pt_path = os.path.join(preprocessed_dir, f"{split}.pt")
            torch.save(split_lists[split], pt_path)
            log.info(f"Done processing split '{split}'. Saved in '{pt_path}'.")


if __name__ == "__main__":
    fire.Fire(preprocess_geomdrugs)
