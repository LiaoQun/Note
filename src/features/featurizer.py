import json
import os
from collections import defaultdict
from typing import Dict, List, Union

from rdkit import Chem
from rdkit.Chem.rdchem import Atom, Bond


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
            self._load_from_json(vocab_filepath)

    def _load_from_json(self, filepath: str):
        """Loads vocabularies from a JSON file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Vocabulary file not found at: {filepath}")

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
