from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data import Data

from src.features.base import BaseFeaturizer
from src.features.chem_prop_featurizer import (
    MultiHotAtomFeaturizer,
    MultiHotBondFeaturizer,
)


class ChemPropPyGFeaturizer(BaseFeaturizer):
    """
    An adapter that uses Chemprop's featurizers (atom and bond) and converts
    the output into a PyTorch Geometric `Data` object.
    """

    def __init__(self, atom_featurizer_mode: str = "v2"):
        """
        Initializes the featurizer.

        Args:
            atom_featurizer_mode (str): The mode for the atom featurizer.
                                        Supported: 'v1', 'v2', 'organic'.
        """
        if atom_featurizer_mode == "v1":
            self.atom_featurizer = MultiHotAtomFeaturizer.v1()
        elif atom_featurizer_mode == "v2":
            self.atom_featurizer = MultiHotAtomFeaturizer.v2()
        elif atom_featurizer_mode == "organic":
            self.atom_featurizer = MultiHotAtomFeaturizer.organic()
        else:
            raise ValueError(f"Unsupported atom featurizer mode: {atom_featurizer_mode}")

        self.bond_featurizer = MultiHotBondFeaturizer()

    @property
    def atom_dim(self) -> int:
        return len(self.atom_featurizer)

    @property
    def bond_dim(self) -> int:
        return len(self.bond_featurizer)

    @property
    def is_discrete(self) -> bool:
        # Chemprop featurizers produce continuous (one-hot) vectors
        return False

    def featurize(
        self,
        mol: Chem.Mol,
        labels: Optional[Dict[Tuple[int, int], float]] = None,
        smiles: str = "",
    ) -> Optional[Data]:
        """
        Featurizes the molecule into a PyG Data object.

        Args:
            mol (Chem.Mol): RDKit molecule with hydrogens.
            labels (Optional[Dict...]]): Optional BDE labels for training.
            smiles (str): Canonical SMILES string.

        Returns:
            Optional[Data]: A PyG Data object or None if the molecule has no bonds.
        """
        # Atom features (V)
        atom_features = [self.atom_featurizer(atom) for atom in mol.GetAtoms()]
        x = torch.from_numpy(np.array(atom_features)).float()

        # Bond features (E) and edge_index
        edge_indices, edge_attrs, bond_indices_map = [], [], []
        
        is_training = labels is not None
        edge_bde_labels, edge_masks = [], []

        for bond in mol.GetBonds():
            u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bond_feat = self.bond_featurizer(bond)
            
            # Add forward and backward edges
            edge_indices.extend([(u, v), (v, u)])
            edge_attrs.extend([bond_feat, bond_feat])
            bond_indices_map.extend([bond.GetIdx(), bond.GetIdx()])

            if is_training:
                canonical_bond_key = tuple(sorted((u, v)))
                bde_label = labels.get(canonical_bond_key)
                
                # Assign labels to both directed edges
                if bde_label is not None:
                    edge_bde_labels.extend([bde_label, bde_label])
                    edge_masks.extend([True, True])
                else:
                    edge_bde_labels.extend([0.0, 0.0])
                    edge_masks.extend([False, False])
        
        if not edge_indices:
            return None

        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.from_numpy(np.array(edge_attrs)).float()

        # Create PyG Data object
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        data.bond_indices_map = torch.tensor(bond_indices_map, dtype=torch.long)
        data.original_input_smiles = smiles
        data.is_valid = torch.tensor(True, dtype=torch.bool) # Assume valid if processed

        if is_training:
            data.y = torch.tensor(edge_bde_labels, dtype=torch.float)
            data.mask = torch.tensor(edge_masks, dtype=torch.bool)

        return data
