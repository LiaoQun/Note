"""
Main script for training and evaluating the BDE Prediction Model.
"""
import os
import shutil
import argparse
import json
from datetime import datetime
import pandas as pd
import torch
from torch_geometric.data import DataLoader
from rdkit import Chem
from tqdm import tqdm
from typing import List, Tuple, Dict
from sklearn.model_selection import train_test_split

from src.config import MainConfig
from src.features.featurizer import Tokenizer
from src.data.dataset import BDEDataset
from src.models.mpnn import BDEModel
from src.training.trainer import Trainer


def load_and_merge_data(data_paths: List[str]) -> pd.DataFrame:
    """
    Loads data from a list of CSV file paths, merges them, canonicalizes SMILES,
    and cleans the data.

    Args:
        data_paths (List[str]): A list of file paths to the CSV data.

    Returns:
        pd.DataFrame: A single, cleaned DataFrame containing all the data.
    """
    if not data_paths:
        raise ValueError("No data paths provided in the configuration.")

    df_list = []
    print("Loading data from the following paths:")
    for path in data_paths:
        if os.path.exists(path):
            print(f" - Loading {path}...")
            try:
                df_list.append(pd.read_csv(path))
            except Exception as e:
                print(f"Warning: Could not read file {path}. Error: {e}. Skipping.")
        else:
            print(f"Warning: Data file not found at: {path}. Skipping.")
    
    if not df_list:
        raise FileNotFoundError("No valid data files could be loaded from the specified paths.")

    print("\nMerging and cleaning data...")
    merged_df = pd.concat(df_list, ignore_index=True)
    print(f"Total records loaded: {len(merged_df)}")

    # Handle missing values
    initial_rows = len(merged_df)
    merged_df.dropna(subset=['molecule', 'bond_index', 'bde'], inplace=True)
    if initial_rows > len(merged_df):
        print(f"Dropped {initial_rows - len(merged_df)} rows with missing key values (molecule, bond_index, or bde).")

    # --- Canonicalize SMILES ---
    print("Canonicalizing SMILES strings...")
    
    def canonicalize(smi):
        try:
            mol = Chem.MolFromSmiles(smi)
            return Chem.MolToSmiles(mol, canonical=True) if mol else None
        except Exception:
            return None

    initial_rows = len(merged_df)
    merged_df['molecule'] = merged_df['molecule'].apply(canonicalize)
    merged_df.dropna(subset=['molecule'], inplace=True)
    if initial_rows > len(merged_df):
        print(f"Dropped {initial_rows - len(merged_df)} rows due to invalid/unparsable SMILES strings.")

    # --- Handle duplicates ---
    # First pass: drop duplicates after loading
    initial_rows = len(merged_df)
    merged_df.drop_duplicates(subset=['molecule', 'bond_index'], keep='first', inplace=True)
    if initial_rows > len(merged_df):
        print(f"Dropped {initial_rows - len(merged_df)} duplicate records (based on molecule and bond_index).")

    print(f"Final cleaned dataset contains {len(merged_df)} records.")
    return merged_df



def prepare_data(df: pd.DataFrame) -> List[Tuple[str, Dict[Tuple[int, int], float]]]:
    """
    Processes a DataFrame into a list of (SMILES, bde_labels_dict) tuples.
    """
    smiles_data = []
    grouped = df.groupby('molecule')
    unique_smiles = list(grouped.groups.keys())

    print(f"Preparing data for {len(unique_smiles)} molecules...")
    for smiles in tqdm(unique_smiles, desc="Processing molecules"):
        mol_df = grouped.get_group(smiles)
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        mol = Chem.AddHs(mol)

        bde_labels_dict = {}
        for _, row in mol_df.iterrows():
            bond_idx = int(row['bond_index'])
            bde = float(row['bde'])
            
            try:
                if bond_idx >= mol.GetNumBonds():
                    continue
                bond = mol.GetBondWithIdx(bond_idx)
                u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                canonical_bond_key = (min(u, v), max(u, v))
                bde_labels_dict[canonical_bond_key] = bde
            except Exception:
                pass
                
        smiles_data.append((smiles, bde_labels_dict))
        
    return smiles_data

def run_training(cfg: MainConfig, config_path: str):
    """
    Main function to set up and run the training and evaluation pipeline.
    """
    # 1. Setup
    torch.manual_seed(cfg.data.random_seed)
    device = torch.device(cfg.train.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create a unique directory for this run
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(cfg.train.output_dir, run_timestamp)
    os.makedirs(run_dir, exist_ok=True)
    print(f"Saving all artifacts to: {run_dir}")

    # Save the config file for this run for reproducibility
    shutil.copy(config_path, os.path.join(run_dir, 'config.json'))
    print(f"Saved configuration to {run_dir}")
    
    # 2. Load, Merge, and Clean Data
    df = load_and_merge_data(cfg.data.data_paths)

    if df.empty:
        print("Stopping run: No data available after loading and cleaning.")
        return

    if 0 < cfg.data.sample_percentage < 1.0:
        print(f"Sampling {cfg.data.sample_percentage * 100:.2f}% of unique molecules...")
        unique_mols = df['molecule'].unique()
        n_mols = max(1, int(len(unique_mols) * cfg.data.sample_percentage))
        sampled_mols = pd.Series(unique_mols).sample(n=n_mols, random_state=cfg.data.random_seed)
        df = df[df['molecule'].isin(sampled_mols)]
        print(f"Dataset reduced to {len(df['molecule'].unique())} unique molecules and {len(df)} entries.")
    
    smiles_data = prepare_data(df)
    
    print("Splitting data...")
    train_val_data, test_data = train_test_split(smiles_data, test_size=cfg.data.test_size, random_state=cfg.data.random_seed)
    val_split_ratio = cfg.data.val_size / (1.0 - cfg.data.test_size)
    train_data, val_data = train_test_split(train_val_data, test_size=val_split_ratio, random_state=cfg.data.random_seed)

    print(f"Training set: {len(train_data)} | Validation set: {len(val_data)} | Test set: {len(test_data)}")

    # 3. Initialize Tokenizer, Datasets, and DataLoaders
    print("Initializing tokenizer...")
    if cfg.data.vocab_path and os.path.exists(cfg.data.vocab_path):
        print(f"Loading tokenizer from predefined vocabulary: {cfg.data.vocab_path}")
        tokenizer = Tokenizer(vocab_filepath=cfg.data.vocab_path)
        effective_vocab_path = cfg.data.vocab_path
    else:
        print("No valid vocabulary path provided. Building tokenizer from training data...")
        # Extract SMILES from the training set to build the vocab
        train_smiles = [data[0] for data in train_data]
        
        tokenizer = Tokenizer()
        tokenizer.build_from_smiles(train_smiles)
        
        # Save the newly created vocab to the run-specific directory
        vocab_save_path = os.path.join(run_dir, "vocab.json")
        tokenizer.save(vocab_save_path)
        print(f"New vocabulary saved to: {vocab_save_path}")
        effective_vocab_path = vocab_save_path

    print("Initializing datasets...")
    train_dataset = BDEDataset(root=os.path.join(cfg.data.dataset_dir, 'train'), smiles_data=train_data, tokenizer=tokenizer)
    val_dataset = BDEDataset(root=os.path.join(cfg.data.dataset_dir, 'val'), smiles_data=val_data, tokenizer=tokenizer)
    test_dataset = BDEDataset(root=os.path.join(cfg.data.dataset_dir, 'test'), smiles_data=test_data, tokenizer=tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.train.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=cfg.train.batch_size, shuffle=False)
    
    # 4. Initialize Model and Optimizer
    print("Initializing model...")
    model = BDEModel(
        num_atom_classes=tokenizer.atom_num_classes + 1,
        num_bond_classes=tokenizer.bond_num_classes + 1,
        atom_features=cfg.model.atom_features,
        num_messages=cfg.model.num_messages
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)

    # 5. Initialize and run Trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        cfg=cfg.train,
        run_dir=run_dir,
        # Pass additional data for final evaluation and saving
        full_dataset_df=df,
        data_splits={'train': train_data, 'val': val_data, 'test': test_data},
        vocab_path=effective_vocab_path
    )
    
    trainer.train()
    trainer.evaluate()


def main():
    parser = argparse.ArgumentParser(description="Train BDE Prediction Model from a config file.")
    parser.add_argument(
        '--config_path',
        type=str,
        default="config.json",
        help='Path to the JSON configuration file.'
    )
    args = parser.parse_args()

    # Initialize base config with defaults
    config = MainConfig()

    # Load and merge config from JSON file
    if os.path.exists(args.config_path):
        print(f"Loading configuration from {args.config_path}...")
        with open(args.config_path, 'r') as f:
            try:
                json_config = json.load(f)
                for group, params in json_config.items():
                    if hasattr(config, group):
                        config_group = getattr(config, group)
                        for key, value in params.items():
                            if hasattr(config_group, key):
                                setattr(config_group, key, value)
                            else:
                                print(f"Warning: Unknown parameter '{key}' in group '{group}' found in JSON. Skipping.")
                    else:
                        print(f"Warning: Unknown config group '{group}' found in JSON. Skipping.")
            except json.JSONDecodeError:
                print(f"Error: Invalid JSON in config file: {args.config_path}. Using defaults.")
    else:
        print(f"Info: Config file not found at '{args.config_path}'. Using default settings.")


    try:
        run_training(config, args.config_path)
    finally:
        # Cleanup
        if os.path.exists(config.data.dataset_dir):
            print(f"Cleaning up temporary dataset directory: {config.data.dataset_dir}")
            shutil.rmtree(config.data.dataset_dir)

if __name__ == '__main__':
    main()
