from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data import Data

from src.features.base import BaseFeaturizer
from src.features.chem_prop_atom import get_multi_hot_atom_featurizer

class ChemPropFeaturizer(BaseFeaturizer):
    """
    Featurizer that uses ChemProp's Multi-Hot atom features and simple bond features.
    Produces float tensors for node features instead of integer indices.
    """
    def __init__(self, mode: str = 'V2'):
        self.atom_featurizer = get_multi_hot_atom_featurizer(mode)
        # We need a bond vocabulary or similar mechanism.
        # Since ChemProp also uses vector features for bonds, we should ideally implement that.
        # For this prototype, we will implement a simple one-hot bond featurizer here directly.
        
        # Simple Bond Features:
        # BondType (Single, Double, Triple, Aromatic) = 4
        # Conjugated (bool) = 1
        # InRing (bool) = 1
        # Stereo (None, Any, Z, E, Cis, Trans) - simplified to just "is STEREO" for now? 
        # Let's keep it simple: 4 (type) + 1 (conjugated) + 1 (ring) = 6 dims
        self._bond_dim = 6 

    @property
    def atom_dim(self) -> int:
        return len(self.atom_featurizer)

    @property
    def bond_dim(self) -> int:
        return self._bond_dim

    @property
    def is_discrete(self) -> bool:
        return False

    def _featurize_bond(self, bond: Chem.Bond) -> List[float]:
        """Simple one-hot encoding for bonds."""
        # 1. Bond Type
        bt = bond.GetBondType()
        bond_feats = [
            1.0 if bt == Chem.rdchem.BondType.SINGLE else 0.0,
            1.0 if bt == Chem.rdchem.BondType.DOUBLE else 0.0,
            1.0 if bt == Chem.rdchem.BondType.TRIPLE else 0.0,
            1.0 if bt == Chem.rdchem.BondType.AROMATIC else 0.0,
        ]
        # 2. Conjugated
        bond_feats.append(1.0 if bond.GetIsConjugated() else 0.0)
        # 3. Ring
        bond_feats.append(1.0 if bond.IsInRing() else 0.0)
        
        return bond_feats

    def featurize(self, mol: Chem.Mol, 
                  labels: Optional[Dict[Tuple[int, int], float]] = None,
                  smiles: str = "") -> Optional[Data]:
        
        # 1. Atom Features
        atom_features = []
        for atom in mol.GetAtoms():
            # Returns np.ndarray
            atom_features.append(self.atom_featurizer(atom))
        
        # Optimize: Convert list of arrays to single numpy array first
        x = torch.FloatTensor(np.array(atom_features)) # [num_atoms, atom_dim]
        
        # 2. Edge Features
        is_training = labels is not None
        edge_indices, edge_attrs, bond_indices_map = [], [], []
        edge_bde_labels, edge_masks = [], []

        for bond in mol.GetBonds():
            u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            bond_feat = self._featurize_bond(bond)
            
            # Add forward and backward edges
            for (start_atom, end_atom) in [(u, v), (v, u)]:
                edge_indices.append((start_atom, end_atom))
                edge_attrs.append(bond_feat)
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
        edge_attr = torch.FloatTensor(edge_attrs) # [num_edges, bond_dim]
        
        # 3. Create Data Object
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        
        if is_training:
            data.y = torch.FloatTensor(edge_bde_labels)
            data.mask = torch.BoolTensor(edge_masks)
            
        data.bond_indices_map = torch.LongTensor(bond_indices_map)
        data.original_input_smiles = smiles
        # Assume valid if we got this far
        data.is_valid = torch.tensor(True, dtype=torch.bool)
        
        return data
