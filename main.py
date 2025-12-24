"""
Main script for training and evaluating the BDE Prediction Model.
"""
import os
import shutil
import argparse
import json
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from rdkit import Chem
from tqdm import tqdm
from typing import List, Tuple, Dict, Any
from sklearn.model_selection import train_test_split

from src.config import MainConfig, DataConfig, ModelConfig, TrainConfig
from src.features.featurizer import Tokenizer
from src.data.dataset import BDEDataset
from src.models.mpnn import BDEModel

def prepare_data(df: pd.DataFrame) -> List[Tuple[str, Dict[Tuple[int, int], float]]]:
    """
    Processes a DataFrame into a list of (SMILES, bde_labels_dict) tuples.
    This function is copied from the original run/train.py script.
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

def run_training(cfg: MainConfig):
    """
    Main function to run the training and evaluation pipeline.
    """
    # 1. Setup
    torch.manual_seed(cfg.data.random_seed)
    device = torch.device(cfg.train.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 2. Load and Prepare Data
    print(f"Loading raw data from {cfg.data.data_path}...")
    if not os.path.exists(cfg.data.data_path):
        raise FileNotFoundError(f"Data file not found at: {cfg.data.data_path}")
    df = pd.read_csv(cfg.data.data_path)

    # Apply sampling if specified
    if 0 < cfg.data.sample_percentage < 1.0:
        print(f"Sampling {cfg.data.sample_percentage * 100:.2f}% of unique molecules...")
        unique_mols = df['molecule'].unique()
        num_mols_to_sample = max(1, int(len(unique_mols) * cfg.data.sample_percentage))
        
        # Sample unique molecules to ensure all BDEs for a molecule are kept together
        sampled_mols = pd.Series(unique_mols).sample(
            n=num_mols_to_sample, random_state=cfg.data.random_seed
        )
        df = df[df['molecule'].isin(sampled_mols)]
        print(f"Dataset reduced to {len(df['molecule'].unique())} unique molecules and {len(df)} entries.")
    elif cfg.data.sample_percentage == 1.0:
        print("Using full dataset.")
    else:
        raise ValueError("sample_percentage must be between 0 (exclusive) and 1 (inclusive).")
    
    smiles_data = prepare_data(df)
    
    print("Splitting data into training, validation, and test sets...")
    train_val_data, test_data = train_test_split(smiles_data, test_size=cfg.data.test_size, random_state=cfg.data.random_seed)
    
    val_split_ratio = cfg.data.val_size / (1.0 - cfg.data.test_size)
    train_data, val_data = train_test_split(train_val_data, test_size=val_split_ratio, random_state=cfg.data.random_seed)

    print(f"Training set size: {len(train_data)}")
    print(f"Validation set size: {len(val_data)}")
    print(f"Test set size: {len(test_data)}")

    # 3. Initialize Tokenizer, Datasets, and DataLoaders
    print("Initializing tokenizer and datasets...")
    if not os.path.exists(cfg.data.vocab_path):
        raise FileNotFoundError(f"Vocabulary file not found at: {cfg.data.vocab_path}")
    tokenizer = Tokenizer(vocab_filepath=cfg.data.vocab_path)
    
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

    # 5. Training Loop
    print("Starting training...")
    best_val_loss = float('inf')
    for epoch in range(1, cfg.train.epochs + 1):
        model.train()
        total_train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False):
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch)
            if batch.mask.sum() > 0:
                loss = F.l1_loss(pred[batch.mask], batch.y[batch.mask])
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item() * batch.num_graphs
        
        avg_train_loss = total_train_loss / len(train_loader.dataset) if len(train_loader.dataset) > 0 else 0

        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch} [Val]", leave=False):
                batch = batch.to(device)
                pred = model(batch)
                if batch.mask.sum() > 0:
                    loss = F.l1_loss(pred[batch.mask], batch.y[batch.mask])
                    total_val_loss += loss.item() * batch.num_graphs
        
        avg_val_loss = total_val_loss / len(val_loader.dataset) if len(val_loader.dataset) > 0 else 0
        
        print(f"Epoch {epoch:03d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), cfg.train.model_save_path)
            print(f"  -> New best validation loss: {best_val_loss:.4f}. Model saved to {cfg.train.model_save_path}")

    print("\nTraining finished.")
    
    # 6. Final Test Evaluation
    print(f"\nLoading best model from {cfg.train.model_save_path} and evaluating on test set...")
    model.load_state_dict(torch.load(cfg.train.model_save_path))
    model.eval()
    total_test_loss = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="[Test]"):
            batch = batch.to(device)
            pred = model(batch)
            if batch.mask.sum() > 0:
                loss = F.l1_loss(pred[batch.mask], batch.y[batch.mask])
                total_test_loss += loss.item() * batch.num_graphs
    
    avg_test_loss = total_test_loss / len(test_loader.dataset) if len(test_loader.dataset) > 0 else 0
    print(f"Final Test MAE: {avg_test_loss:.4f} kcal/mol")

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
    try:
        print(f"Loading configuration from {args.config_path}...")
        with open(args.config_path, 'r') as f:
            json_config = json.load(f)
        
        # Update the default config with values from the JSON file
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

    except FileNotFoundError:
        print(f"Info: Config file not found at '{args.config_path}'. Using default settings.")
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in config file: {args.config_path}. Aborting.")
        return

    try:
        run_training(config)
    finally:
        # 7. Cleanup
        if os.path.exists(config.data.dataset_dir):
            print(f"Cleaning up temporary dataset directory: {config.data.dataset_dir}")
            shutil.rmtree(config.data.dataset_dir)

if __name__ == '__main__':
    main()
