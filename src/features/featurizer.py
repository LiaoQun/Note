import json
import os
from typing import Dict, List, Optional, Tuple, Union

import torch
from rdkit import Chem
from rdkit.Chem.rdchem import Atom, Bond
from torch_geometric.data import Data

from src.features.base import BaseFeaturizer


def get_ring_size(mol_obj: Union[Atom, Bond], max_size: int = 6) -> int:
    """
    Helper to determine the size of the smallest ring an atom or bond is in.

    Args:
        mol_obj (Union[Atom, Bond]): The RDKit Atom or Bond object.
        max_size (int): The maximum ring size to consider. Larger rings will be capped.

    Returns:
        int: The size of the smallest ring the object is in, or 0 if not in a ring.
             Returns `max_size` if the smallest ring is larger than `max_size`.
    """
    if not mol_obj.IsInRing():
        return 0

    min_ring_size = float('inf')
    for ring in mol_obj.GetOwningMol().GetRingInfo().AtomRings():
        if mol_obj.GetIdx() in ring:
            min_ring_size = min(min_ring_size, len(ring))

    if min_ring_size <= max_size:
        return min_ring_size
    else:
        return max_size


def atom_featurizer(atom: Atom) -> str:
    """
    Generates a feature string for an RDKit Atom object.

    Args:
        atom (Atom): The RDKit Atom object.

    Returns:
        str: A string representation of the atom's features.
    """
    return str(
        (
            atom.GetSymbol(),
            atom.GetNumRadicalElectrons(),
            atom.GetFormalCharge(),
            atom.GetChiralTag(),
            atom.GetIsAromatic(),
            get_ring_size(atom, max_size=6),
            atom.GetDegree(),
            atom.GetTotalNumHs(includeNeighbors=True),
        )
    )


def bond_featurizer(bond: Bond, flipped: bool = False) -> str:
    """
    Generates a feature string for an RDKit Bond object.

    Args:
        bond (Bond): The RDKit Bond object.
        flipped (bool): If True, reverses the begin/end atom order in the feature string.

    Returns:
        str: A string representation of the bond's features.
    """
    if not flipped:
        atoms = "{}-{}".format(
            bond.GetBeginAtom().GetSymbol(), bond.GetEndAtom().GetSymbol()
        )
    else:
        atoms = "{}-{}".format(
            bond.GetEndAtom().GetSymbol(), bond.GetBeginAtom().GetSymbol()
        )

    btype = str((bond.GetBondType(), bond.GetIsConjugated()))
    ring = f"R{get_ring_size(bond, max_size=6)}" if bond.IsInRing() else ""

    return " ".join([atoms, btype, ring]).strip()


class Tokenizer:
    """
    Handles the mapping of feature strings to integer IDs for atoms and bonds.
    Supports loading from a predefined vocabulary and dynamic vocabulary expansion.
    """

    def __init__(self, vocab_filepath: str = None):
        """
        Initializes the Tokenizer.

        Args:
            vocab_filepath (str, optional): Path to a JSON vocabulary file to load.
        """
        self._atom_vocab: Dict[str, int] = {"unk": 1}
        self._bond_vocab: Dict[str, int] = {"unk": 1}
        self.atom_num_classes: int = 1
        self.bond_num_classes: int = 1

        if vocab_filepath:
            self.load(vocab_filepath)

    def load(self, filepath: str):
        """Loads vocabularies from a JSON file."""
        if not os.path.exists(filepath):
            # It's okay if file doesn't exist on init, we might build it later.
            # But if explicitly called, we warn or raise.
            print(f"Warning: Vocabulary file not found at: {filepath}. Starting with empty vocab.")
            return

        with open(filepath, 'r') as f:
            data = json.load(f)

        if "atom_tokenizer" in data and "_data" in data["atom_tokenizer"]:
            self._atom_vocab = data["atom_tokenizer"]["_data"]
            self.atom_num_classes = data["atom_tokenizer"].get("num_classes", max(self._atom_vocab.values()))

        if "bond_tokenizer" in data and "_data" in data["bond_tokenizer"]:
            self._bond_vocab = data["bond_tokenizer"]["_data"]
            self.bond_num_classes = data["bond_tokenizer"].get("num_classes", max(self._bond_vocab.values()))

    def _add_feature(self, feature_string: str, feature_type: str) -> int:
        """Adds a new feature to the appropriate vocabulary if it doesn't exist."""
        vocab = self._atom_vocab if feature_type == 'atom' else self._bond_vocab
        if feature_string not in vocab:
            if feature_type == 'atom':
                self.atom_num_classes += 1
                vocab[feature_string] = self.atom_num_classes
            else:
                self.bond_num_classes += 1
                vocab[feature_string] = self.bond_num_classes
        return vocab[feature_string]

    def tokenize_atom(self, atom_feature_string: str) -> int:
        """Returns the ID for an atom feature string, defaulting to 'unk'."""
        return self._atom_vocab.get(atom_feature_string, self._atom_vocab["unk"])

    def tokenize_bond(self, bond_feature_string: str) -> int:
        """Returns the ID for a bond feature string, defaulting to 'unk'."""
        return self._bond_vocab.get(bond_feature_string, self._bond_vocab["unk"])

    def build_from_smiles(self, smiles_list: List[str]):
        """Builds or expands the vocabulary from a list of SMILES strings."""
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue  # Or raise an error

            mol = Chem.AddHs(mol)

            for atom in mol.GetAtoms():
                self._add_feature(atom_featurizer(atom), 'atom')

            for bond in mol.GetBonds():
                self._add_feature(bond_featurizer(bond, flipped=False), 'bond')
                self._add_feature(bond_featurizer(bond, flipped=True), 'bond')

    def save(self, filepath: str):
        """Saves the current vocabularies to a JSON file."""
        data = {
            "atom_tokenizer": {"_data": self._atom_vocab, "num_classes": self.atom_num_classes},
            "bond_tokenizer": {"_data": self._bond_vocab, "num_classes": self.bond_num_classes}
        }
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)


class TokenFeaturizer(BaseFeaturizer):
    """
    Featurizer that uses a dictionary (Tokenizer) to map unique string representations
    of atoms and bonds to integer IDs.
    """
    def __init__(self, vocab_filepath: str = None):
        self.tokenizer = Tokenizer(vocab_filepath)

    @property
    def atom_dim(self) -> int:
        return self.tokenizer.atom_num_classes + 1

    @property
    def bond_dim(self) -> int:
        return self.tokenizer.bond_num_classes + 1

    @property
    def is_discrete(self) -> bool:
        return True

    def prepare_data(self, smiles_list: List[str]):
        """Builds vocabulary from a list of SMILES."""
        self.tokenizer.build_from_smiles(smiles_list)

    def save(self, filepath: str):
        self.tokenizer.save(filepath)

    def load(self, filepath: str):
        self.tokenizer.load(filepath)

    def featurize(self, mol: Chem.Mol, 
                  labels: Optional[Dict[Tuple[int, int], float]] = None,
                  smiles: str = "") -> Optional[Data]:
        """
        Converts a single RDKit Mol object into a PyG Data object.
        """
        # 1. Atom Features and Validity
        atom_feature_strings = [atom_featurizer(mol_atom) for mol_atom in mol.GetAtoms()]
        x = torch.LongTensor([self.tokenizer.tokenize_atom(s) for s in atom_feature_strings])
        atoms_are_valid = (x != 1).all().item()

        # 2. Edge Features, BDE Labels, and Validity
        is_training = labels is not None
        edge_indices, edge_attrs, bond_indices_map = [], [], []
        edge_bde_labels, edge_masks = [], []

        for bond in mol.GetBonds():
            u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            
            # Add forward and backward edges
            for (start_atom, end_atom, is_flipped) in [(u, v, False), (v, u, True)]:
                edge_indices.append((start_atom, end_atom))
                edge_attrs.append(self.tokenizer.tokenize_bond(bond_featurizer(bond, flipped=is_flipped)))
                bond_indices_map.append(bond.GetIdx())
                
                if is_training:
                    canonical_bond_key = tuple(sorted((u, v)))
                    bde_label = labels.get(canonical_bond_key)
                    
                    if bde_label is not None:
                        edge_bde_labels.append(bde_label)
                        edge_masks.append(True)
                    else:
                        edge_bde_labels.append(0.0)
                        edge_masks.append(False)

        if not edge_indices:
            return None

        edge_index = torch.LongTensor(edge_indices).t().contiguous()
        edge_attr = torch.LongTensor(edge_attrs)
        bonds_are_valid = (edge_attr != 1).all().item()
        
        # 3. Create Data Object
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        
        # Attach training-specific attributes if available
        if is_training:
            data.y = torch.FloatTensor(edge_bde_labels)
            data.mask = torch.BoolTensor(edge_masks)
            
        # Attach inference-specific and common attributes
        data.bond_indices_map = torch.LongTensor(bond_indices_map)
        data.original_input_smiles = smiles
    
        # Ensure 'is_valid' is set to True if molecule could be processed, else False
        data.is_valid = torch.tensor(atoms_are_valid and bonds_are_valid, dtype=torch.bool)
        
        return data
