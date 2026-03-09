import json
import os
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem.rdchem import Atom, Bond, HybridizationType, BondType
from torch_geometric.data import Data

from src.features.base import BaseFeaturizer

def get_mols_from_smiles(smiles_list: List[str]) -> List[Chem.Mol]:
    """Safely convert a list of SMILES to RDKit Mol objects, skipping invalid ones."""
    mols = []
    for smiles in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f"RDKit failed to parse SMILES: {smiles}")
            mols.append(Chem.AddHs(mol))
        except ValueError:
            # Optionally log the error here
            continue
    return mols

class DynamicMultiHotAtomFeaturizer:
    """Atom featurizer that dynamically builds a multi-hot encoding from a dataset."""
    def __init__(self):
        self.atomic_nums: Dict[int, int] = {}
        self.degrees: Dict[int, int] = {}
        self.formal_charges: Dict[int, int] = {}
        self.chiral_tags: Dict[int, int] = {}
        self.num_Hs: Dict[int, int] = {}
        self.hybridizations: Dict[HybridizationType, int] = {}
        self._subfeats: List[Dict] = []
        self._size = 0
        self._is_built = False

    def build_from_mols(self, mols: List[Chem.Mol]):
        """Builds the feature vocabulary from a list of RDKit molecules."""
        # Use sets to find unique feature values
        atomic_nums_set, degrees_set, charges_set, chirals_set, num_hs_set, hybs_set = set(), set(), set(), set(), set(), set()

        for mol in mols:
            for a in mol.GetAtoms():
                atomic_nums_set.add(a.GetAtomicNum())
                degrees_set.add(a.GetTotalDegree())
                charges_set.add(a.GetFormalCharge())
                chirals_set.add(int(a.GetChiralTag()))
                num_hs_set.add(int(a.GetTotalNumHs()))
                hybs_set.add(a.GetHybridization())

        # Create sorted mappings for consistent encoding
        self.atomic_nums = {val: i for i, val in enumerate(sorted(list(atomic_nums_set)))}
        self.degrees = {val: i for i, val in enumerate(sorted(list(degrees_set)))}
        self.formal_charges = {val: i for i, val in enumerate(sorted(list(charges_set)))}
        self.chiral_tags = {val: i for i, val in enumerate(sorted(list(chirals_set)))}
        self.num_Hs = {val: i for i, val in enumerate(sorted(list(num_hs_set)))}
        self.hybridizations = {val: i for i, val in enumerate(sorted(list(hybs_set), key=lambda h: str(h)))}

        self._subfeats = [self.atomic_nums, self.degrees, self.formal_charges, self.chiral_tags, self.num_Hs, self.hybridizations]
        self._size = sum(len(choices) + 1 for choices in self._subfeats) + 2  # +1 for unknown, +2 for IsAromatic/Mass
        self._is_built = True

    def __len__(self) -> int:
        return self._size if self._is_built else 2

    def __call__(self, a: Atom) -> np.ndarray:
        if not self._is_built:
            raise RuntimeError("Featurizer has not been built. Call `prepare_data` first.")
            
        x = np.zeros(self._size)
        feats = [a.GetAtomicNum(), a.GetTotalDegree(), a.GetFormalCharge(), int(a.GetChiralTag()), int(a.GetTotalNumHs()), a.GetHybridization()]
        
        i = 0
        for feat, choices in zip(feats, self._subfeats):
            j = choices.get(feat, len(choices))
            x[i + j] = 1
            i += len(choices) + 1
            
        x[i] = int(a.GetIsAromatic())
        x[i + 1] = 0.01 * a.GetMass()
        return x

    def get_state(self) -> Dict:
        return {
            "atomic_nums": self.atomic_nums,
            "degrees": self.degrees,
            "formal_charges": self.formal_charges,
            "chiral_tags": self.chiral_tags,
            "num_Hs": self.num_Hs,
            "hybridizations": {str(k).split('.')[-1]: v for k, v in self.hybridizations.items()},
        }

    def set_state(self, state: Dict):
        self.atomic_nums = {int(k): v for k, v in state["atomic_nums"].items()}
        self.degrees = {int(k): v for k, v in state["degrees"].items()}
        self.formal_charges = {int(k): v for k, v in state["formal_charges"].items()}
        self.chiral_tags = {int(k): v for k, v in state["chiral_tags"].items()}
        self.num_Hs = {int(k): v for k, v in state["num_Hs"].items()}
        self.hybridizations = {getattr(HybridizationType, k): v for k, v in state.get("hybridizations", {}).items()}
        
        self._subfeats = [self.atomic_nums, self.degrees, self.formal_charges, self.chiral_tags, self.num_Hs, self.hybridizations]
        self._size = sum(len(choices) + 1 for choices in self._subfeats) + 2
        self._is_built = True

class DynamicMultiHotBondFeaturizer:
    """Bond featurizer that dynamically builds a multi-hot encoding from a dataset."""
    def __init__(self):
        self.bond_types: Dict[BondType, int] = {}
        self.stereos: Dict[int, int] = {}
        self._size = 0
        self._is_built = False

    def build_from_mols(self, mols: List[Chem.Mol]):
        bond_types_set, stereos_set = set(), set()
        for mol in mols:
            for b in mol.GetBonds():
                bond_types_set.add(b.GetBondType())
                stereos_set.add(int(b.GetStereo()))

        self.bond_types = {bt: i for i, bt in enumerate(sorted(list(bond_types_set), key=lambda x: str(x)))}
        self.stereos = {s: i for i, s in enumerate(sorted(list(stereos_set)))}
        self._size = 1 + len(self.bond_types) + 2 + (len(self.stereos) + 1)
        self._is_built = True

    def __len__(self) -> int:
        return self._size if self._is_built else 4

    def __call__(self, b: Bond) -> np.ndarray:
        if not self._is_built:
            raise RuntimeError("Featurizer has not been built. Call `prepare_data` first.")
        x = np.zeros(self._size, dtype=float)
        if b is None:
            x[0] = 1
            return x
        i = 1
        bt_bit = self.bond_types.get(b.GetBondType())
        if bt_bit is not None:
            x[i + bt_bit] = 1
        i += len(self.bond_types)
        x[i], i = float(b.GetIsConjugated()), i + 1
        x[i], i = float(b.IsInRing()), i + 1
        stereo_bit = self.stereos.get(int(b.GetStereo()))
        if stereo_bit is not None:
            x[i + stereo_bit] = 1
        else:
            x[i + len(self.stereos)] = 1
        return x

    def get_state(self) -> Dict:
        return {
            "bond_types": {str(k).split('.')[-1]: v for k, v in self.bond_types.items()},
            "stereos": self.stereos,
        }

    def set_state(self, state: Dict):
        self.bond_types = {getattr(BondType, k): v for k, v in state["bond_types"].items()}
        self.stereos = {int(k): v for k, v in state["stereos"].items()}
        self._size = 1 + len(self.bond_types) + 2 + (len(self.stereos) + 1)
        self._is_built = True

class ChemPropFeaturizer(BaseFeaturizer):
    """
    Implements a dynamic ChemProp-style multi-hot featurizer that learns from data.
    """
    def __init__(self, vocab_filepath: str = None):
        self.atom_featurizer = DynamicMultiHotAtomFeaturizer()
        self.bond_featurizer = DynamicMultiHotBondFeaturizer()
        if vocab_filepath:
            self.load(vocab_filepath)

    @property
    def atom_dim(self) -> int: return len(self.atom_featurizer)
    @property
    def bond_dim(self) -> int: return len(self.bond_featurizer)
    @property
    def is_discrete(self) -> bool: return False

    def prepare_data(self, smiles_list: List[str]):
        """Builds feature vocabulary from a list of SMILES."""
        print("Building ChemProp featurizer from SMILES list...")
        mols = get_mols_from_smiles(smiles_list)
        self.atom_featurizer.build_from_mols(mols)
        self.bond_featurizer.build_from_mols(mols)
        print("Build complete.")

    def save(self, filepath: str):
        """Saves the learned feature state to a JSON file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump({
                "atom_featurizer": self.atom_featurizer.get_state(),
                "bond_featurizer": self.bond_featurizer.get_state(),
            }, f, indent=4)

    def load(self, filepath: str):
        """Loads the feature state from a JSON file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Vocab file not found at: {filepath}")
        with open(filepath, 'r') as f:
            state = json.load(f)
        self.atom_featurizer.set_state(state["atom_featurizer"])
        self.bond_featurizer.set_state(state["bond_featurizer"])

    def featurize(self, mol: Chem.Mol, labels: Optional[Dict[Tuple[int, int], float]] = None, smiles: str = "") -> Optional[Data]:
        atom_features = [self.atom_featurizer(atom) for atom in mol.GetAtoms()]
        if not atom_features: return None
        x = torch.from_numpy(np.array(atom_features)).float()

        edge_indices, edge_attrs, bond_indices_map = [], [], []
        is_training = labels is not None
        edge_bde_labels, edge_masks = [], []

        for bond in mol.GetBonds():
            u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bond_feat = self.bond_featurizer(bond)
            edge_indices.extend([(u, v), (v, u)])
            edge_attrs.extend([bond_feat, bond_feat])
            bond_indices_map.extend([bond.GetIdx(), bond.GetIdx()])
            if is_training:
                key = tuple(sorted((u, v)))
                label = labels.get(key)
                edge_bde_labels.extend([label, label] if label is not None else [0.0, 0.0])
                edge_masks.extend([label is not None, label is not None])
        
        if not edge_indices: return None

        data = Data(x=x, edge_index=torch.tensor(edge_indices, dtype=torch.long).t().contiguous(), 
                    edge_attr=torch.from_numpy(np.array(edge_attrs)).float())
        data.bond_indices_map = torch.tensor(bond_indices_map, dtype=torch.long)
        data.original_input_smiles = smiles
        data.is_valid = torch.tensor(True, dtype=torch.bool)
        if is_training:
            data.y = torch.tensor(edge_bde_labels, dtype=torch.float)
            data.mask = torch.tensor(edge_masks, dtype=torch.bool)
        return data
