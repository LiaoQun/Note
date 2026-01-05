"""
This module contains the Predictor class for running inference with a trained BDE model.
"""
import os
from typing import List, Dict, Union

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from torch_geometric.data import Data, Batch

from src.data_preparation.template_generator import generate_fragment_template
from src.features.featurizer import Tokenizer, atom_featurizer, bond_featurizer
from src.models.mpnn import BDEModel


def _featurize_mol_for_inference(mol: Chem.Mol, tokenizer: Tokenizer, canonical_smiles: str) -> Data:
    """
    Converts a single RDKit Mol object into a PyG Data object for inference.
    Crucially, it creates a mapping from PyG's internal edge order back to RDKit's original bond indices.
    
    Args:
        mol (Chem.Mol): The input RDKit molecule, with hydrogens added.
        tokenizer (Tokenizer): The tokenizer instance loaded from the training vocabulary.
        canonical_smiles (str): The canonical SMILES string of the molecule.

    Returns:
        Data: A PyG Data object ready for the model.
    """
    # Atom features
    atom_feature_strings = [atom_featurizer(atom) for atom in mol.GetAtoms()]
    x = torch.LongTensor([tokenizer.tokenize_atom(s) for s in atom_feature_strings])

    # Edge features
    edge_indices = []
    edge_attrs = []
    # bond_indices_map is essential for mapping model predictions back to original RDKit bonds.
    # It stores the RDKit bond index for each directed edge in the order they are added.
    bond_indices_map = [] # To map model output back to original bond indices

    for bond in mol.GetBonds():
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        
        # Forward edge (u -> v)
        edge_indices.append((u, v))
        edge_attrs.append(tokenizer.tokenize_bond(bond_featurizer(bond, flipped=False)))
        # Map this directed edge back to the original RDKit bond index
        bond_indices_map.append(bond.GetIdx())
        
        # Backward edge (v -> u)
        edge_indices.append((v, u))
        edge_attrs.append(tokenizer.tokenize_bond(bond_featurizer(bond, flipped=True)))
        # Map this directed edge back to the same original RDKit bond index
        bond_indices_map.append(bond.GetIdx())

    if not edge_indices:
        # Handle cases where molecule has no bonds (e.g., single atom)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0,), dtype=torch.long)
    else:
        edge_index = torch.LongTensor(edge_indices).t().contiguous()
        edge_attr = torch.LongTensor(edge_attrs)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.bond_indices_map = torch.LongTensor(bond_indices_map) # Store the mapping in the Data object
    data.original_input_smiles = canonical_smiles # Store canonical smiles for mapping back
    
    return data


class Predictor:
    """Handles loading a trained model and making BDE predictions."""

    def __init__(self, model_path: str, vocab_path: str, device: str = 'cpu'):
        """
        Initializes the Predictor.

        Args:
            model_path (str): Path to the trained model checkpoint (.pt file).
            vocab_path (str): Path to the vocabulary file (.json) used during training.
            device (str): The device to run inference on ('cpu' or 'cuda').
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"Vocabulary file not found at: {vocab_path}")

        self.tokenizer = Tokenizer(vocab_filepath=vocab_path)
        self.device = torch.device(device)

        # Re-create model architecture based on vocab size
        self.model = BDEModel(
            num_atom_classes=self.tokenizer.atom_num_classes + 1,
            num_bond_classes=self.tokenizer.bond_num_classes + 1,
        ).to(self.device)
        
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        print("Model and tokenizer loaded successfully.")

    def predict(self, smiles_list: List[str], drop_duplicates: bool = True) -> pd.DataFrame:
        """
        Runs BDE prediction for a list of molecules in batch.

        Args:
            smiles_list (List[str]): A list of SMILES strings.
            drop_duplicates (bool): If True, remove predictions for bonds that
                                    result in the same set of fragments.

        Returns:
            pd.DataFrame: A concatenated DataFrame with predictions for all molecules.
        """
        all_data_list = []
        all_fragments_df_list = []
        
        # 1. Generate fragment info for all molecules and collect Data objects
        for original_smiles_input_idx, smiles in enumerate(smiles_list):
            fragment_df = generate_fragment_template([smiles])
            if fragment_df.empty:
                print(f"No valid bonds found for SMILES '{smiles}'. Skipping.")
                continue
            
            canonical_smiles = fragment_df['molecule'].iloc[0] # Canonical SMILES used
            
            mol = Chem.MolFromSmiles(canonical_smiles)
            mol = Chem.AddHs(mol)
            
            data = _featurize_mol_for_inference(mol, self.tokenizer, canonical_smiles)
            
            all_data_list.append(data)
            all_fragments_df_list.append(fragment_df)

        if not all_data_list:
            return pd.DataFrame()

        # Concatenate all fragment DataFrames
        all_fragments_df = pd.concat(all_fragments_df_list, ignore_index=True)

        # 2. Batch featurized molecules and run model inference (single call)
        batch = Batch.from_data_list(all_data_list).to(self.device)
        
        with torch.no_grad():
            raw_predictions = self.model(batch) # Tensor of predictions for all edges in batch

        # 3. Map predictions back to original molecule and bond indices
        preds_records = []
        start_edge_idx = 0
        for i, data in enumerate(all_data_list):
            num_edges = data.edge_index.size(1) # Number of edges for this specific graph
            end_edge_idx = start_edge_idx + num_edges

            # Extract raw predictions for the current molecule's graph
            graph_preds = raw_predictions[start_edge_idx:end_edge_idx]
            # Retrieve the bond_indices_map created during featurization for this graph
            graph_bond_indices = data.bond_indices_map.cpu().numpy()

            # Create a temporary DataFrame to associate predictions with their original RDKit bond indices
            graph_preds_df = pd.DataFrame({
                'molecule': data.original_input_smiles,
                'bond_index': graph_bond_indices,
                'bde_pred': graph_preds.cpu().numpy()
            })
            
            # Group by molecule and bond_index to average predictions for the two directed edges
            # (forward and backward) that correspond to a single original RDKit bond.
            bde_preds_by_bond = graph_preds_df.groupby(['molecule', 'bond_index'])['bde_pred'].mean().reset_index()
            preds_records.append(bde_preds_by_bond)
            
            start_edge_idx = end_edge_idx
            
        final_bde_preds = pd.concat(preds_records, ignore_index=True) if preds_records else pd.DataFrame()

        # 4. Merge the averaged BDE predictions with the detailed fragment information
        # This step ensures the predicted BDE is correctly associated with its fragments.
        result_df = pd.merge(all_fragments_df, final_bde_preds, on=['molecule', 'bond_index'], how='left')

        # Drop the now-redundant bde column from the template if exists
        if 'bde' in result_df.columns:
            result_df = result_df.drop(columns=['bde'])

        if drop_duplicates:
            # Sort fragments within each row to create a canonical key for deduplication
            fragments = result_df[['fragment1', 'fragment2']].values
            canonical_frag_pairs = [tuple(sorted(f)) for f in fragments]
            
            # Add a temporary column for deduplication
            temp_dedup_df = result_df.copy()
            temp_dedup_df['canonical_frag_pair'] = canonical_frag_pairs
            
            result_df = temp_dedup_df.drop_duplicates(subset=['molecule', 'canonical_frag_pair']).drop(columns=['canonical_frag_pair'])
            result_df = result_df.reset_index(drop=True)

        return result_df.reset_index(drop=True)


def get_bde_predictions(
    smiles: Union[str, List[str]],
    model_path: str,
    vocab_path: str,
    drop_duplicates: bool = True,
    device: str = 'cpu'
) -> pd.DataFrame:
    """
    A simple, one-shot function to get BDE predictions for one or more molecules.

    This is a high-level wrapper around the Predictor class that handles
    initialization, prediction, and result formatting.

    Args:
        smiles (Union[str, List[str]]): A single SMILES string or a list of SMILES strings.
        model_path (str): Path to the trained model checkpoint (.pt file).
        vocab_path (str): Path to the vocabulary file (.json) used during training.
        drop_duplicates (bool, optional): If True, remove predictions for bonds that
                                          result in the same set of fragments. Defaults to True.
        device (str, optional): The device to run inference on ('cpu' or 'cuda'). Defaults to 'cpu'.

    Returns:
        pd.DataFrame: A DataFrame containing predictions and fragment info.
    """
    if isinstance(smiles, str):
        smiles_list = [smiles]
    else:
        smiles_list = smiles

    try:
        predictor = Predictor(
            model_path=model_path,
            vocab_path=vocab_path,
            device=device
        )
        results_df = predictor.predict(smiles_list, drop_duplicates=drop_duplicates)
        return results_df
    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return pd.DataFrame()