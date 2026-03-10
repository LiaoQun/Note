from typing import Dict, List, Optional, Tuple
import math

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

    def __init__(self, atom_featurizer: MultiHotAtomFeaturizer, bond_featurizer: MultiHotBondFeaturizer):
        self.atom_featurizer = atom_featurizer
        self.bond_featurizer = bond_featurizer

    @classmethod
    def from_smiles(cls, smiles_list: List[str], save_path: str) -> "ChemPropPyGFeaturizer":
        """從訓練資料動態建構，並序列化到 save_path。"""
        atom_feat = MultiHotAtomFeaturizer.from_smiles(smiles_list)
        bond_feat = MultiHotBondFeaturizer()
        instance = cls(atom_feat, bond_feat)
        instance._save(save_path)
        return instance

    @classmethod
    def from_vocab(cls, vocab_path: str) -> "ChemPropPyGFeaturizer":
        """從序列化檔案重建，保證與訓練時行為一致。"""
        import json
        from rdkit.Chem.rdchem import HybridizationType

        with open(vocab_path, 'r') as f:
            data = json.load(f)

        # 將 int 還原為 HybridizationType enum
        hybridization_map = {h.real: h for h in HybridizationType.values.values()}
        hybridizations = [hybridization_map[v] for v in data["hybridizations"]]

        atom_feat = MultiHotAtomFeaturizer(
            atomic_nums=data["atomic_nums"],
            degrees=data["degrees"],
            formal_charges=data["formal_charges"],
            chiral_tags=data["chiral_tags"],
            num_Hs=data["num_Hs"],
            hybridizations=hybridizations,
        )
        return cls(atom_feat, MultiHotBondFeaturizer())

    def _save(self, save_path: str):
        import json, os
        from rdkit.Chem.rdchem import HybridizationType

        atom_feat = self.atom_featurizer
        # 從 atom_featurizer 反推出原始參數列表
        data = {
            "featurizer_type": "ChemPropFeaturizer",
            "atomic_nums": sorted(atom_feat.atomic_nums.keys()),
            "degrees": sorted(atom_feat.degrees.keys()),
            "formal_charges": sorted(
                atom_feat.formal_charges.keys(),
                key=lambda x: list(atom_feat.formal_charges.keys()).index(x)
            ),
            "chiral_tags": sorted(atom_feat.chiral_tags.keys()),
            "num_Hs": sorted(atom_feat.num_Hs.keys()),
            "hybridizations": [h.real for h in atom_feat.hybridizations.keys()],
        }
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=4)

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
        labels: Optional[Dict[Tuple[int, int], List[float]]] = None,
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
                bde_labels = labels.get(canonical_bond_key)
                
                # Assign labels to both directed edges
                if bde_labels is not None:
                    # Convert NaNs to 0 internally, and use mask to ignore them during loss computation
                    cleaned_labels = [float(lbl) if not math.isnan(lbl) else 0.0 for lbl in bde_labels]
                    masks = [not math.isnan(lbl) for lbl in bde_labels]
                    
                    edge_bde_labels.extend([cleaned_labels, cleaned_labels])
                    edge_masks.extend([masks, masks])
                else:
                    # Assume 1 task dimension if labels is empty just to avoid breaking during inference init test
                    num_tasks = len(list(labels.values())[0]) if len(labels) > 0 else 1
                    zero_labels = [0.0] * num_tasks
                    false_masks = [False] * num_tasks
                    
                    edge_bde_labels.extend([zero_labels, zero_labels])
                    edge_masks.extend([false_masks, false_masks])
        
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
