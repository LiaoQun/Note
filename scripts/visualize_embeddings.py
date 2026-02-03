"""
Script for visualizing bond embeddings from intermediate layers of a trained BDEModel.
"""
import argparse
import json
import os
import sys
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from rdkit import Chem
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from torch_geometric.data import DataLoader
from tqdm import tqdm

from src.config import MainConfig
from src.data.dataset import BDEDataset
from src.features import get_featurizer
from src.models.mpnn import BDEModel

def featurize_bond_for_plotting(smiles: str, bond_index: int) -> str:
    """Generates a human-readable string for a bond's type and environment.
    Handles RDKit parsing errors gracefully.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return "Invalid_SMILES"
        
        # Ensure bond_index is int
        bond_index = int(bond_index)

        # Validate bond_index range
        if bond_index < 0 or bond_index >= mol.GetNumBonds():
            return "Invalid_Bond_Index_Range"
        
        bond = mol.GetBondWithIdx(bond_index)
        
        atom1_symbol = bond.GetBeginAtom().GetSymbol()
        atom2_symbol = bond.GetEndAtom().GetSymbol()
        
        # Sort symbols alphabetically to make bond types canonical (e.g., C-H not H-C)
        symbols = sorted([atom1_symbol, atom2_symbol])
        return f"{symbols[0]}-{symbols[1]}"
    except Exception as e:
        # Catch any other RDKit or general processing errors
        return f"Error_{type(e).__name__}"

def main():
    parser = argparse.ArgumentParser(description="Visualize bond embeddings from a trained BDE model.")
    parser.add_argument('--run_dir', type=str, required=True, help='Path to the training run directory.')
    parser.add_argument('--data_path', type=str, default='examples/test_data.csv.gz', help='Path to the data file for sampling.')
    parser.add_argument('--num_samples', type=int, default=2000, help='Number of bonds to sample for visualization.')
    parser.add_argument('--output_path', type=str, default=None, help='Path to save the output plot SVG file.')
    args = parser.parse_args()

    # --- 1. Load Config and Model ---
    print(f"Loading model and config from {args.run_dir}...")
    config_path = os.path.join(args.run_dir, 'config.json')
    with open(config_path, 'r') as f:
        run_config_dict = json.load(f)

    # Create a MainConfig object from the dict
    cfg = MainConfig()
    cfg.data.featurizer_type = run_config_dict.get('data', {}).get('featurizer_type', 'TokenFeaturizer')
    cfg.model.atom_features = run_config_dict.get('model', {}).get('atom_features', 128)
    cfg.model.num_messages = run_config_dict.get('model', {}).get('num_messages', 6)
    
    # Initialize Featurizer
    featurizer = get_featurizer(cfg.data)

    # Initialize Model
    model = BDEModel(
        atom_input_dim=featurizer.atom_dim,
        bond_input_dim=featurizer.bond_dim,
        atom_features=cfg.model.atom_features,
        num_messages=cfg.model.num_messages,
        inputs_are_discrete=featurizer.is_discrete
    )
    
    # Load state dict
    model_path = os.path.join(args.run_dir, run_config_dict.get('train', {}).get('model_save_path', 'bde_model.pt'))
    state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
    # Handle backward compatibility for key names
    key_mapping = {
        "atom_embedding.weight": "atom_encoder.weight", "bond_embedding.weight": "bond_encoder.weight",
        "bond_mean_embedding.weight": "bond_bias_encoder.weight"
    }
    new_state_dict = {key_mapping.get(k, k): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.eval()
    print("Model loaded successfully.")

    # --- 2. Load and Sample Data ---
    print(f"Loading and sampling data from {args.data_path}...")
    df = pd.read_csv(args.data_path)
    df.dropna(subset=['molecule', 'bond_index', 'bde'], inplace=True)
    
    # Sample bonds, not just molecules
    if len(df) > args.num_samples:
        df_sample = df.sample(n=args.num_samples, random_state=42)
    else:
        df_sample = df
        
    smiles_data_for_dataset = [] # List of (smiles, bde_labels_dict) for BDEDataset
    bond_labels_raw = [] # Raw labels collected here for featurize_bond_for_plotting
    
    # Pre-process SMILES data to filter out RDKit errors and prepare for BDEDataset
    for smiles_idx, (smiles, group) in enumerate(tqdm(df_sample.groupby('molecule'), desc="Pre-processing SMILES for Dataset")):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue

            bde_labels_dict = {}
            for row_idx, row in group.iterrows():
                bond_idx = int(row.bond_index)
                if bond_idx < 0 or bond_idx >= mol.GetNumBonds():
                    continue
                
                bond = mol.GetBondWithIdx(bond_idx)
                canonical_bond_key = tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())))
                bde_labels_dict[canonical_bond_key] = row.bde
                
                # Collect raw labels for later plotting, using the robust featurizer_bond_for_plotting
                # Now collect labels for both forward and backward edges to match embeddings
                bond_labels_raw.append({'smiles': smiles, 'bond_index': bond_idx})
                bond_labels_raw.append({'smiles': smiles, 'bond_index': bond_idx}) # Duplicate for backward edge


            if bde_labels_dict: # Only add if there are valid bonds
                smiles_data_for_dataset.append((smiles, bde_labels_dict))
        
        except Exception as e:
            continue

    if not smiles_data_for_dataset:
        print("No valid SMILES data left after pre-processing. Aborting.")
        return

    dataset = BDEDataset(root='temp_vis_dataset', smiles_data=smiles_data_for_dataset, featurizer=featurizer)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    # --- 3. Extract Intermediate Embeddings ---
    print("Extracting intermediate embeddings...")
    all_embeddings: Dict[str, List[torch.Tensor]] = {}
    bond_labels_for_plotting_final = [] # Final labels after matching with actual extracted embeddings

    with torch.no_grad():
        for batch in tqdm(loader, desc="Forward pass"):
            intermediate_embeds = model(batch, return_intermediate=True)
            
            for layer_name, embed_tensor in intermediate_embeds.items():
                if layer_name not in all_embeddings:
                    all_embeddings[layer_name] = []
                all_embeddings[layer_name].append(embed_tensor.cpu())

            mol_smiles_list_in_batch = [s for s in batch.original_input_smiles] # Get original smiles from batch
            bond_indices_map = batch.bond_indices_map.cpu().numpy()
            
            # This logic needs to carefully reconstruct bond labels from the batch
            # The order must match the embeddings exactly
            
            # Reconstruct labels for ALL edges in the batch
            num_edges_in_batch = batch.edge_index.size(1)
            for edge_idx_in_batch in range(num_edges_in_batch):
                # Determine which molecule this edge belongs to
                mol_idx_in_batch = batch.batch[batch.edge_index[0, edge_idx_in_batch]].item()
                smiles_for_this_bond = mol_smiles_list_in_batch[mol_idx_in_batch]
                original_rdkit_bond_idx = bond_indices_map[edge_idx_in_batch]
                
                bond_labels_for_plotting_final.append(featurize_bond_for_plotting(smiles_for_this_bond, original_rdkit_bond_idx))

    # Concatenate all tensors
    for layer_name, tensor_list in all_embeddings.items():
        all_embeddings[layer_name] = torch.cat(tensor_list, dim=0)
    
    # --- 4. Dimensionality Reduction ---
    print("Performing dimensionality reduction (PCA + t-SNE)...")
    pipeline = Pipeline([
        ('pca', PCA(n_components=50, random_state=42)),
        ('tsne', TSNE(n_components=2, random_state=42, perplexity=30))
    ])
    
    transformed_embeddings: Dict[str, np.ndarray] = {}
    for layer_name, embed_tensor in all_embeddings.items():
        print(f"  - Reducing {layer_name}...")
        transformed_embeddings[layer_name] = pipeline.fit_transform(embed_tensor.numpy())

    # --- 5. Plotting ---
    print("Generating plot...")
    
    # Filter out 'Error' and 'Invalid' labels before plotting
    valid_labels_mask = np.array([not (label.startswith("Invalid") or label.startswith("Error")) for label in bond_labels_for_plotting_final])
    
    # Apply mask to transformed embeddings
    filtered_transformed_embeddings: Dict[str, np.ndarray] = {}
    for layer_name, coords in transformed_embeddings.items():
        filtered_transformed_embeddings[layer_name] = coords[valid_labels_mask]

    # Apply mask to labels
    filtered_bond_labels = np.array(bond_labels_for_plotting_final)[valid_labels_mask]

    if len(filtered_bond_labels) == 0:
        print("No valid bonds to plot after filtering. Aborting plot generation.")
        return
        
    label_series = pd.Series(filtered_bond_labels)
    common_labels = label_series.value_counts().nlargest(10).index.tolist()
    
    plot_data = {'label': filtered_bond_labels}
    for layer_name, coords in filtered_transformed_embeddings.items():
        plot_data[f'{layer_name}_x'] = coords[:, 0]
        plot_data[f'{layer_name}_y'] = coords[:, 1]
    
    plot_df = pd.DataFrame(plot_data)
    plot_df = plot_df[plot_df['label'].isin(common_labels)]

    if plot_df.empty:
        print("No data points for common labels after filtering. Aborting plot generation.")
        return

    num_layers = len(filtered_transformed_embeddings)
    fig, axes = plt.subplots(1, num_layers, figsize=(5 * num_layers, 5), sharex=True, sharey=True)
    if num_layers == 1: axes = [axes]

    sns.set_style("whitegrid")
    palette = sns.color_palette("husl", len(common_labels))
    
    layer_names = sorted(filtered_transformed_embeddings.keys(), key=lambda x: int(x.split('_')[-1]))

    for i, layer_name in enumerate(layer_names):
        ax = axes[i]
        sns.scatterplot(
            data=plot_df, x=f'{layer_name}_x', y=f'{layer_name}_y', hue='label',
            palette=palette, ax=ax, s=10, alpha=0.7, legend=(i == num_layers -1)
        )
        ax.set_title(layer_name.replace('_', ' ').title())
        ax.set_xlabel('')
        ax.set_ylabel('')

    if num_layers > 1:
        # Move legend outside the plot
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., title='Bond Type', markerscale=2)
    
    plt.tight_layout()
    plt.subplots_adjust(right=0.7) # Adjust to make space for legend

    output_file = args.output_path or os.path.join(args.run_dir, 'embedding_visualization.svg')
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")
    
    import shutil
    if os.path.exists('temp_vis_dataset'):
        shutil.rmtree('temp_vis_dataset')

if __name__ == '__main__':
    main()