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
from src.features.featurizer import Tokenizer, mol_to_graph
from src.models.mpnn import BDEModel


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
        
        # The warning for weights_only=False is a security feature.
        # It's safe to set weights_only=True here as we are only loading model parameters.
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
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
            
            # Use the shared graph converter
            data = mol_to_graph(mol, self.tokenizer, canonical_smiles)
            
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
                'bde_pred': graph_preds.cpu().numpy(),
                'is_valid': data.is_valid.item() # Add the is_valid flag for the molecule
            })
            
            # Group by molecule and bond_index to average predictions for the two directed edges
            # (forward and backward) that correspond to a single original RDKit bond.
            # The 'is_valid' flag will be the same for all bonds of a molecule, so mean() is fine.
            bde_preds_by_bond = graph_preds_df.groupby(['molecule', 'bond_index'])[['bde_pred', 'is_valid']].mean().reset_index()
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