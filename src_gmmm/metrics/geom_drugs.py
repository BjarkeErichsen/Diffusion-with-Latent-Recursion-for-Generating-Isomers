from pathlib import Path
from typing import Optional

import ase
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from rdkit.Geometry import Point3D

from ..data.utils import read_json
from ..metrics.base import Metrics, discrete_histogram
from .geom_drugs_utils import compute_molecules_stability, geom_drugs_h_tuple_valencies

_SYMBOLS_GEOMDRUGS = ["H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Br", "I", "Bi"]


class GeomDrugsMetrics(Metrics):
    def __init__(
        self,
        atom_types_str: list[str] = _SYMBOLS_GEOMDRUGS,
        max_num_atoms: int = 40,
        json_path: Optional[str | Path] = None,
        summarize_hidden: bool = False,
        hidden_prefix: str = "_",
        remove_h: bool = False,
    ):
        if remove_h:
            if "H" in atom_types_str:
                h_idx = atom_types_str.index("H")
                atom_types_str = [s for s in atom_types_str if s != "H"]
            else:
                h_idx = None
        else:
            h_idx = None

        self.remove_h = remove_h
        self.encoder = {s: idx for idx, s in enumerate(atom_types_str)}
        self.max_num_atoms = max_num_atoms

        if json_path:
            dataset_infos = read_json(json_path=json_path)
            ref_smiles = set(dataset_infos.get("smiles", []))
            ref_atom_hist = np.array(dataset_infos.get("atom_hist")) if dataset_infos.get("atom_hist") is not None else None
            if remove_h and h_idx is not None and ref_atom_hist is not None:
                ref_atom_hist = np.delete(ref_atom_hist, h_idx)
                if ref_atom_hist.sum() > 0:
                    ref_atom_hist = ref_atom_hist / ref_atom_hist.sum()
        else:
            ref_smiles = set([])
            ref_atom_hist = None

        self.ref_smiles = ref_smiles
        self.ref_atom_hist = ref_atom_hist

        self.summarize_hidden = summarize_hidden
        self.hidden_prefix = hidden_prefix

        self.reset()

    def reset(self):
        self.smiles = []
        self.valid = []
        self.valid_connected = []
        self.connected = []
        self.n_atoms = []
        self.total_atoms_with_hs = []
        self.molecule_stable = []
        self.atom_stable = []
        self.atom_hist = []

    def update(self, atoms: list[ase.Atoms] | ase.Atoms):
        if isinstance(atoms, ase.Atoms):
            atoms = [atoms]

        for a in atoms:
            if self.remove_h:
                atom_indices = [i for i, sym in enumerate(a.symbols) if sym != 'H']
            else:
                atom_indices = list(range(len(a.symbols)))
                
            atom_symbols = [a.symbols[i] for i in atom_indices]
            atom_positions = a.positions[atom_indices]
            n_heavy = len([sym for sym in atom_symbols if sym != 'H'])

            v = 0
            c = 0
            is_connected = 0
            molecule_stable = 0
            atom_stable = 0
            total_atoms = 0
            smi = None
            n_atoms_current = len(atom_symbols)

            if n_atoms_current > 0:
                # 1. create object with atomic positions
                mol = Chem.RWMol()
                for symbol in atom_symbols:
                    mol.AddAtom(Chem.Atom(symbol))
                conf = Chem.Conformer(n_atoms_current)
                for idx, pos in enumerate(atom_positions):
                    conf.SetAtomPosition(idx, Point3D(float(pos[0]), float(pos[1]), float(pos[2])))
                mol.AddConformer(conf)

                try:
                    # 2. Use func rdDetermineBonds.DetermineBonds
                    rd_success = False
                    for chg in [0, 1, -1, 2, -2]:
                        try:
                            temp_mol = Chem.Mol(mol)
                            rdDetermineBonds.DetermineBonds(temp_mol, charge=chg)
                            Chem.SanitizeMol(temp_mol)
                            mol = temp_mol
                            rd_success = True
                            break
                        except Exception:
                            continue
                    
                    if not rd_success:
                        raise ValueError("Failed to determine bonds with any charge")

                    if self.remove_h:
                        # 3. Let it create hydrogens itself
                        mol_eval = Chem.AddHs(mol, addCoords=True, explicitOnly=True)
                    else:
                        mol_eval = mol

                    is_connected = int(len(Chem.GetMolFrags(mol_eval)) == 1)

                    # 4. Use the exact function from isayevlab to determine stability & validity
                    # is_valid already sanitizes and checks for single fragment
                    val, stab, stab_atoms, atom_counts = compute_molecules_stability(
                        [mol_eval], aromatic=True, allowed_bonds=geom_drugs_h_tuple_valencies
                    )
                    v = int(val[0].item())
                    c = v  # Their is_valid function enforces len(GetMolFrags) == 1
                    molecule_stable = int(stab[0].item())
                    atom_stable = int(stab_atoms[0].item())
                    total_atoms = int(atom_counts[0].item())

                    if v == 1:
                        smi = Chem.MolToSmiles(mol_eval, canonical=True)
                        
                except Exception:
                    v = 0
                    c = 0
                    is_connected = 0
                    molecule_stable = 0
                    atom_stable = 0
                    total_atoms = n_heavy
                    smi = None

            self.smiles.append(smi)
            self.valid.append(v)
            self.valid_connected.append(c)
            self.connected.append(is_connected)
            self.n_atoms.append(n_heavy)
            self.molecule_stable.append(molecule_stable)
            self.atom_stable.append(atom_stable)
            self.total_atoms_with_hs.append(total_atoms)
            self.atom_hist.append(discrete_histogram(atom_symbols, encoder=self.encoder))

    def summarize(self) -> dict:
        assert len(self.valid) == len(self.valid_connected)
        assert len(self.valid) == len(self.molecule_stable)
        assert len(self.valid) == len(self.smiles)

        n_samples = len(self.valid)
        n_atoms = sum(self.n_atoms)

        summary = {}

        if n_samples == 0:
            return summary

        n_atoms_total = sum(self.total_atoms_with_hs)

        # We cut down the redundant non-hydrogen metrics and now rely entirely on the exact PyTorch-based valency table metric.
        summary["atom_stable"] = sum(self.atom_stable) / max(n_atoms_total, 1)
        summary["molecule_stable"] = sum(self.molecule_stable) / n_samples
        summary["valid_connected"] = sum(self.valid_connected) / n_samples
        summary["connected"] = sum(self.connected) / n_samples

        valid_unique_smiles = set(
            [smiles for (v, smiles) in zip(self.valid, self.smiles) if v and smiles is not None]
        )
        summary["valid_unique"] = len(valid_unique_smiles) / n_samples
        if self.ref_smiles is not None:
            vun_smiles = valid_unique_smiles.difference(self.ref_smiles)
            summary["valid_unique_novel"] = len(vun_smiles) / n_samples

        atom_hist = np.sum(np.stack(self.atom_hist, axis=0), axis=0)
        atom_hist_sum = atom_hist.sum()
        if atom_hist_sum > 0:
            atom_hist = atom_hist / atom_hist_sum
        if self.ref_atom_hist is not None and self.ref_atom_hist.shape == atom_hist.shape:
            summary["tv_atom"] = np.sum(np.abs(self.ref_atom_hist - atom_hist)).item()

        if self.summarize_hidden:
            summary[f"{self.hidden_prefix}atom_hist"] = atom_hist
            summary[f"{self.hidden_prefix}num_atoms_hist"] = discrete_histogram(
                self.n_atoms,
                encoder={idx: idx for idx in range(self.max_num_atoms + 1)},
                norm=True,
            )
            summary[f"{self.hidden_prefix}smiles"] = list(valid_unique_smiles)

        return summary
