import torch
import warnings
from rdkit import Chem

# Valency table from isayevlab/geom-drugs-3dgen-evaluation/blob/main/scripts/compute_molecule_stability.py
geom_drugs_h_tuple_valencies = {
    "Br": {0: [(0, 1)], 1: [(0, 2)]},
    "C": {0: [(0, 4), (2, 2), (2, 1), (3, 0)], -1: [(0, 3), (2, 1), (3, 0)], 1: [(0, 3), (2, 1), (3, 0)]},
    "N": {0: [(0, 3), (2, 0), (2, 1), (3, 0)], 1: [(0, 4), (2, 0), (2, 1), (2, 2), (3, 0)], -1: [(0, 2), (2, 0)], -2: [(0, 1)]},
    "H": {0: [(0, 1)]},
    "S": {0: [(0, 2), (0, 3), (0, 6), (2, 0)], 1: [(0, 3), (2, 0), (2, 1), (3, 0)], 2: [(0, 4), (2, 1), (2, 2)], 3: [(0, 2), (0, 5)], -1: [(0, 1)]},
    "O": {0: [(0, 2), (2, 0)], -1: [(0, 1)], 1: [(0, 3)]},
    "F": {0: [(0, 1)]},
    "Cl": {0: [(0, 1)], 1: [(0, 2)]},
    "P": {0: [(0, 3), (0, 5)], 1: [(0, 4)]},
    "I": {0: [(0, 1)], 1: [(0, 2)], 2: [(0, 3)]},
    "Si": {0: [(0, 4)], 1: [(0, 5)]},
    "B": {-1: [(0, 4)], 0: [(0, 3)]},
    "Bi": {0: [(0, 3)], 2: [(0, 5)]}
}

# ==============================================================================
# The following functions (is_valid, _is_valid_valence_tuple, 
# compute_molecules_stability_from_graph, compute_molecules_stability) are 
# copied exactly from the isayevlab/geom-drugs-3dgen-evaluation repository.
# They provide the official PyTorch-based logic for validating valency and 
# aromatic bonds in the GEOM-Drugs dataset.
# ==============================================================================

def is_valid(mol, verbose=False):
    if mol is None:
        return False
    try:
        Chem.SanitizeMol(mol)
    except Chem.rdchem.KekulizeException as e:
        if verbose:
            print(f"Kekulization failed: {e}")
        return False
    except ValueError as e:
        if verbose:
            print(f"Sanitization failed: {e}")
        return False
    if len(Chem.GetMolFrags(mol)) > 1:
        if verbose:
            print("Molecule has multiple fragments.")
        return False
    return True

def _is_valid_valence_tuple(combo, allowed, charge, element_symbol=None):
    if isinstance(allowed, tuple):
        return combo == allowed
    elif isinstance(allowed, (list, set)):
        return combo in allowed
    elif isinstance(allowed, dict):
        if charge not in allowed:
            element_info = f" for element {element_symbol}" if element_symbol else ""
            warnings.warn(
                f"Missing charge state {charge}{element_info} in valency table. "
                f"Available charges: {list(allowed.keys())}. Assuming invalid.",
                UserWarning
            )
            return False
        return _is_valid_valence_tuple(combo, allowed[charge], charge, element_symbol)
    elif allowed == []:
        element_info = f" for element {element_symbol}" if element_symbol else ""
        warnings.warn(
            f"Empty valency configuration{element_info} with charge {charge}. Assuming invalid.",
            UserWarning
        )
        return False
    return False

def compute_molecules_stability_from_graph(adjacency_matrices, numbers, charges, allowed_bonds=None, aromatic=True):
    if adjacency_matrices.ndim == 2:
        adjacency_matrices = adjacency_matrices.unsqueeze(0)
        numbers = numbers.unsqueeze(0)
        charges = charges.unsqueeze(0)
    if allowed_bonds is None:
        allowed_bonds = geom_drugs_h_tuple_valencies
    if not aromatic:
        assert (adjacency_matrices == 1.5).sum() == 0 and (adjacency_matrices == 4).sum() == 0
    batch_size = adjacency_matrices.shape[0]
    stable_mask = torch.zeros(batch_size)
    n_stable_atoms = torch.zeros(batch_size)
    n_atoms = torch.zeros(batch_size)
    for i in range(batch_size):
        adj = adjacency_matrices[i]
        atom_nums = numbers[i]
        atom_charges = charges[i]
        mol_stable = True
        n_atoms_i, n_stable_i = 0, 0
        for j, (a_num, charge) in enumerate(zip(atom_nums, atom_charges)):
            if a_num.item() == 0:
                continue
            row = adj[j]
            aromatic_count = int((row == 1.5).sum().item())
            normal_valence = float((row * (row != 1.5)).sum().item())
            combo = (aromatic_count, int(normal_valence))
            symbol = Chem.GetPeriodicTable().GetElementSymbol(int(a_num))
            allowed = allowed_bonds.get(symbol, {})
            if _is_valid_valence_tuple(combo, allowed, int(charge), symbol):
                n_stable_i += 1
            else:
                mol_stable = False
            n_atoms_i += 1
        stable_mask[i] = float(mol_stable)
        n_stable_atoms[i] = n_stable_i
        n_atoms[i] = n_atoms_i
    return stable_mask, n_stable_atoms, n_atoms

def compute_molecules_stability(rdkit_molecules, aromatic=True, allowed_bonds=None):
    stable_list, stable_atoms_list, atom_counts_list, validity_list = [], [], [], []
    for mol in rdkit_molecules:
        if mol is None:
            continue
        n_atoms = mol.GetNumAtoms()
        adj = torch.zeros((1, n_atoms, n_atoms))
        numbers = torch.zeros((1, n_atoms), dtype=torch.long)
        charges = torch.zeros((1, n_atoms), dtype=torch.long)
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            numbers[0, idx] = atom.GetAtomicNum()
            charges[0, idx] = atom.GetFormalCharge()
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            bond_type = bond.GetBondTypeAsDouble()
            adj[0, i, j] = adj[0, j, i] = bond_type
        stable, stable_atoms, atom_count = compute_molecules_stability_from_graph(
            adj, numbers, charges, allowed_bonds, aromatic
        )
        stable_list.append(stable.item())
        stable_atoms_list.append(stable_atoms.item())
        atom_counts_list.append(atom_count.item())
        validity_list.append(float(is_valid(mol)))
    return (
        torch.tensor(validity_list),
        torch.tensor(stable_list),
        torch.tensor(stable_atoms_list),
        torch.tensor(atom_counts_list)
    )
