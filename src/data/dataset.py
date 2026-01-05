import os
import json
from typing import Dict, List, Tuple, Union

import torch
from torch_geometric.data import Dataset, Data # Changed from InMemoryDataset
from rdkit import Chem
from rdkit.Chem.rdchem import Bond
from tqdm import tqdm # Added for process method progress bar

from src.features.featurizer import atom_featurizer, bond_featurizer, Tokenizer


class BDEDataset(Dataset): # Inherit from Dataset
    """
    A PyTorch Geometric Dataset for BDE prediction.
    Processes SMILES strings into PyG Data objects with atom and bond features,
    BDE labels, and a loss mask. Each Data object is saved as a separate file,
    allowing for larger-than-memory datasets.
    """
    def __init__(self, root: str, smiles_data: List[Tuple[str, Dict[Tuple[int, int], float]]], tokenizer: Tokenizer, transform=None, pre_transform=None):
        """
        Args:
            root (str): Root directory where the dataset will be saved.
            smiles_data (List[Tuple[str, Dict[Tuple[int, int], float]]]): A list of tuples,
                                                                          each containing a SMILES string
                                                                          and a dictionary of BDE labels.
                                                                          BDE labels are mapped from bond tuple (atom_idx1, atom_idx2) to BDE value.
            tokenizer (Tokenizer): An initialized Tokenizer instance from src.features.featurizer.
        """
        self.smiles_data = smiles_data
        self.tokenizer = tokenizer
        super().__init__(root, transform, pre_transform)
        # self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False) # Removed for Dataset

    @property
    def raw_file_names(self) -> List[str]:
        # No raw files, as data is passed directly
        return []

    @property
    def processed_file_names(self) -> List[str]:
        # Return a list of names for each individual processed graph file
        # This assumes self.smiles_data is known at init time to get the count
        return [f'data_{i}.pt' for i in range(len(self.smiles_data))]

    def download(self):
        # Data is passed directly, so no download needed
        pass

    def process(self):
        """
        Processes SMILES data into PyG Data objects and saves each one as a separate .pt file.
        """
        # Create processed directory if it doesn't exist
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

        for i, (smiles, bde_labels_dict) in enumerate(tqdm(self.smiles_data, desc="Processing SMILES to PyG Data")):
            processed_path = os.path.join(self.processed_dir, f'data_{i}.pt')
            if os.path.exists(processed_path):
                continue # Skip if already processed

            try:
                data = self._mol_to_pyg_data(smiles, bde_labels_dict)
                if data is not None:
                    # Apply pre_transform if specified
                    if self.pre_transform is not None:
                        data = self.pre_transform(data)
                    torch.save(data, processed_path)
            except ValueError as e:
                print(f"Skipping SMILES '{smiles}' (index {i}) due to error: {e}")
                continue
        
        # After processing, apply pre_filter if specified.
        # Note: For Dataset, pre_filter is usually applied when iterating or calling get(),
        # but if we want to filter out files that failed to process, we do it here.
        # For simplicity, we assume process() only saves valid data.

    def len(self) -> int:
        """Returns the number of graphs in the dataset."""
        return len(self.processed_file_names)

    def get(self, idx: int) -> Data:
        """Loads and returns the Data object at the given index."""
        # Ensure the index is within bounds
        if not (0 <= idx < len(self)):
            raise IndexError(f"Index {idx} is out of bounds for dataset of size {len(self)}")
        
        # Construct the path for the specific processed file
        file_path = os.path.join(self.processed_dir, self.processed_file_names[idx])
        
        # Load the Data object
        data = torch.load(file_path)
        
        # Apply transform if specified
        if self.transform is not None:
            data = self.transform(data)
        return data

    def _mol_to_pyg_data(self, smiles: str, bde_labels_dict: Dict[Tuple[int, int], float]) -> Union[Data, None]:
        """
        Converts a SMILES string and its BDE labels into a PyG Data object.

        Args:
            smiles (str): The SMILES string of the molecule.
            bde_labels_dict (Dict[Tuple[int, int], float]): Dictionary mapping
                                                            bond tuples (atom_idx1, atom_idx2)
                                                            to BDE values. Note: bond tuple should be canonical
                                                            (min_idx, max_idx).

        Returns:
            Data: A PyG Data object representing the molecule.
            None: If the molecule cannot be processed (e.g., invalid SMILES).
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            # print(f"Warning: Could not parse SMILES: {smiles}")
            return None
        mol = Chem.AddHs(mol) # Add explicit hydrogens for featurization consistency

        # Node features (x)
        atom_feature_strings = [atom_featurizer(atom) for atom in mol.GetAtoms()]
        x = torch.LongTensor([self.tokenizer.tokenize_atom(s) for s in atom_feature_strings]) # [num_atoms]

        # Edge features (edge_attr, edge_index)
        edge_indices = []
        edge_feature_strings = []
        edge_bde_labels = []
        edge_masks = []

        for bond in mol.GetBonds():
            u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()

            # For BDE labels lookup, use canonical (min_idx, max_idx)
            canonical_bond_key = (min(u, v), max(u, v))
            bde_label = bde_labels_dict.get(canonical_bond_key, None)

            # Original direction (u -> v)
            edge_indices.append([u, v])
            edge_feature_strings.append(bond_featurizer(bond, flipped=False))
            if bde_label is not None:
                edge_bde_labels.append(bde_label)
                edge_masks.append(True)
            else:
                edge_bde_labels.append(0.0) # Placeholder for no label
                edge_masks.append(False)

            # Flipped direction (v -> u)
            edge_indices.append([v, u])
            edge_feature_strings.append(bond_featurizer(bond, flipped=True))
            if bde_label is not None: # BDE label is the same for both directions of the bond
                edge_bde_labels.append(bde_label)
                edge_masks.append(True)
            else:
                edge_bde_labels.append(0.0) # Placeholder for no label
                edge_masks.append(False)
        
        if not edge_indices:
            # Molecule might have no bonds, e.g., a single atom SMILES "C"
            # Return None or a Data object with empty edge lists depending on requirements
            return None # Or raise ValueError, depends on how we want to handle this.
                        # For now, returning None to be filtered out.

        edge_index = torch.LongTensor(edge_indices).T # [2, num_edges]
        edge_attr = torch.LongTensor([self.tokenizer.tokenize_bond(s) for s in edge_feature_strings]) # [num_edges]
        y = torch.FloatTensor(edge_bde_labels) # [num_edges]
        mask = torch.BoolTensor(edge_masks) # [num_edges]

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, mask=mask)
