"Main script for training and evaluating the BDE Prediction Model."
import os
import shutil
import argparse
import json
from datetime import datetime
import pandas as pd
import torch
from torch_geometric.loader import DataLoader
from rdkit import Chem
from tqdm import tqdm
from typing import List, Tuple, Dict
from sklearn.model_selection import train_test_split
import logging # Import logging

from src.config import MainConfig
from src.features import get_featurizer
from src.data.dataset import BDEDataset
from src.models.mpnn import BDEModel # Temporarily keep BDEModel
from src.training.trainer import Trainer

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.StreamHandler(), # Output to console
                        logging.FileHandler("training.log") # Save logs to a file
                    ])
logger = logging.getLogger(__name__) # Get a logger for this module


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
    logger.info("Loading data from the following paths:")
    for path in data_paths:
        if os.path.exists(path):
            logger.info(f" - Loading {path}...")
            try:
                df_list.append(pd.read_csv(path))
            except Exception as e:
                logger.warning(f"Could not read file {path}. Error: {e}. Skipping.", exc_info=True)
        else:
            logger.warning(f"Data file not found at: {path}. Skipping.")
    
    if not df_list:
        raise FileNotFoundError("No valid data files could be loaded from the specified paths.")

    logger.info("\nMerging and cleaning data...")
    merged_df = pd.concat(df_list, ignore_index=True)
    logger.info(f"Total records loaded: {len(merged_df)}")

    # Handle missing values
    initial_rows = len(merged_df)
    merged_df.dropna(subset=['molecule', 'bond_index', 'bde'], inplace=True)
    if initial_rows > len(merged_df):
        logger.info(f"Dropped {initial_rows - len(merged_df)} rows with missing key values (molecule, bond_index, or bde).")

    # --- Canonicalize SMILES ---
    logger.info("Canonicalizing SMILES strings...")
    
    def canonicalize(smi):
        try:
            mol = Chem.MolFromSmiles(smi)
            return Chem.MolToSmiles(mol, canonical=True) if mol else None
        except Exception as e:
            logger.debug(f"Failed to canonicalize SMILES '{smi}': {e}", exc_info=True)
            return None

    initial_rows = len(merged_df)
    merged_df['molecule'] = merged_df['molecule'].apply(canonicalize)
    merged_df.dropna(subset=['molecule'], inplace=True)
    if initial_rows > len(merged_df):
        logger.info(f"Dropped {initial_rows - len(merged_df)} rows due to invalid/unparsable SMILES strings.")

    # --- Handle duplicates ---
    # First pass: drop duplicates after loading
    initial_rows = len(merged_df)
    merged_df.drop_duplicates(subset=['molecule', 'bond_index'], keep='first', inplace=True)
    if initial_rows > len(merged_df):
        logger.info(f"Dropped {initial_rows - len(merged_df)} duplicate records (based on molecule and bond_index).")

    logger.info(f"Final cleaned dataset contains {len(merged_df)} records.")
    return merged_df


def prepare_data(df: pd.DataFrame) -> List[Tuple[str, Dict[Tuple[int, int], float]]]:
    """
    Processes a DataFrame into a list of (SMILES, bde_labels_dict) tuples.
    """
    processed_smiles_data: List[Tuple[str, Dict[Tuple[int, int], float]]] = []
    grouped_df = df.groupby('molecule')
    
    logger.info(f"Preparing BDE labels for {len(grouped_df)} unique molecules...")
    for smiles, mol_df in tqdm(grouped_df, desc="Processing molecules for labels"):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning(f"Skipping molecule '{smiles}' due to RDKit parse error during label preparation.")
            continue
        mol = Chem.AddHs(mol)

        bde_labels_dict = {}
        for _, row in mol_df.iterrows():
            bond_idx = int(row['bond_index'])
            bde = float(row['bde'])
            
            try:
                if bond_idx >= mol.GetNumBonds():
                    logger.warning(f"Bond index {bond_idx} out of range for molecule '{smiles}'. Skipping bond.")
                    continue
                bond = mol.GetBondWithIdx(bond_idx)
                u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                canonical_bond_key = (min(u, v), max(u, v))
                bde_labels_dict[canonical_bond_key] = bde
            except Exception as e:
                logger.warning(f"Error processing bond for {smiles} at bond_index {bond_idx}: {e}", exc_info=True)
                pass
                
        processed_smiles_data.append((smiles, bde_labels_dict))
        
    return processed_smiles_data

def run_training(cfg: MainConfig, config_path: str):
    """
    Main function to set up and run the training and evaluation pipeline.
    """
    # 1. Setup
    torch.manual_seed(cfg.data.random_seed)
    device = torch.device(cfg.train.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Create a unique directory for this run
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(cfg.train.output_dir, run_timestamp)
    os.makedirs(run_dir, exist_ok=True)
    logger.info(f"Saving all artifacts to: {run_dir}")

    # Save the config file for this run for reproducibility
    shutil.copy(config_path, os.path.join(run_dir, 'config.json'))
    logger.info(f"Saved configuration to {run_dir}")
    
    # 2. Load, Merge, and Clean Data
    df = load_and_merge_data(cfg.data.data_paths)

    if df.empty:
        logger.error("Stopping run: No data available after loading and cleaning.")
        return

    if 0 < cfg.data.sample_percentage < 1.0:
        logger.info(f"Sampling {cfg.data.sample_percentage * 100:.2f}% of unique molecules...")
        unique_mols = df['molecule'].unique()
        n_mols = max(1, int(len(unique_mols) * cfg.data.sample_percentage))
        sampled_mols = pd.Series(unique_mols).sample(n=n_mols, random_state=cfg.data.random_seed)
        df = df[df['molecule'].isin(sampled_mols)]
        logger.info(f"Dataset reduced to {len(df['molecule'].unique())} unique molecules and {len(df)} entries.")
    
    processed_smiles_data = prepare_data(df)
    
    logger.info("Splitting data...")
    train_val_smiles_data, test_smiles_data = train_test_split(processed_smiles_data, test_size=cfg.data.test_size, random_state=cfg.data.random_seed)
    val_split_ratio = cfg.data.val_size / (1.0 - cfg.data.test_size)
    train_smiles_data, val_smiles_data = train_test_split(train_val_smiles_data, test_size=val_split_ratio, random_state=cfg.data.random_seed)

    logger.info(f"Initial splits: Train ({len(train_smiles_data)}), Val ({len(val_smiles_data)}), Test ({len(test_smiles_data)}) unique molecule entries.")

    # 3. Initialize Featurizer, Datasets, and DataLoaders
    logger.info(f"Initializing featurizer: {cfg.data.featurizer_type}...")
    
    # Use the factory to get the featurizer based on config
    # Note: If vocab_path is in config, featurizer might load it. 
    # But for new training, we usually want to build it from scratch if it's a TokenFeaturizer.
    featurizer = get_featurizer(cfg.data)

    # Check if we should build vocabulary (only for TokenFeaturizer usually)
    # If vocab_path exists and is valid, the factory might have loaded it.
    # If not, we should build it from training data.
    # BaseFeaturizer has 'prepare_data' hook.
    if hasattr(featurizer, 'prepare_data'):
        logger.info("Preparing featurizer (e.g., building vocabulary)...")
        train_smiles = [data[0] for data in train_smiles_data]
        featurizer.prepare_data(train_smiles)
        
    # Save the featurizer state (e.g., vocab.json) to the run directory
    # For TokenFeaturizer, this is critical. For ChemProp, it might be a no-op.
    # We maintain the legacy 'vocab.json' filename for now if applicable, but better to pass run_dir.
    vocab_save_path = os.path.join(run_dir, "vocab.json")
    featurizer.save(vocab_save_path)
    logger.info(f"Featurizer state saved to: {vocab_save_path}")
    effective_vocab_path = vocab_save_path

    logger.info("Initializing datasets...")
    train_dataset = BDEDataset(root=os.path.join(cfg.data.dataset_dir, 'train'), smiles_data=train_smiles_data, featurizer=featurizer)
    val_dataset = BDEDataset(root=os.path.join(cfg.data.dataset_dir, 'val'), smiles_data=val_smiles_data, featurizer=featurizer)
    test_dataset = BDEDataset(root=os.path.join(cfg.data.dataset_dir, 'test'), smiles_data=test_smiles_data, featurizer=featurizer)
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.train.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=cfg.train.batch_size, shuffle=False)
    
    # 4. Initialize Model and Optimizer
    logger.info("Initializing model...")
    model = BDEModel(
        atom_input_dim=featurizer.atom_dim,
        bond_input_dim=featurizer.bond_dim,
        atom_features=cfg.model.atom_features,
        num_messages=cfg.model.num_messages,
        inputs_are_discrete=featurizer.is_discrete
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
        model_cfg=cfg.model,
        run_dir=run_dir,
        # Pass additional data for final evaluation and saving
        full_dataset_df=df,
        data_splits={'train': train_smiles_data, 'val': val_smiles_data, 'test': test_smiles_data},
        vocab_path=effective_vocab_path, # Used by Predictor
        featurizer_type=cfg.data.featurizer_type
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
        logger.info(f"Loading configuration from {args.config_path}...")
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
                                logger.warning(f"Unknown parameter '{key}' in group '{group}' found in JSON. Skipping.")
                    else:
                        logger.warning(f"Unknown config group '{group}' found in JSON. Skipping.")
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON in config file: {args.config_path}. Using defaults.", exc_info=True)
    else:
        logger.info(f"Config file not found at '{args.config_path}'. Using default settings.")


    try:
        run_training(config, args.config_path)
    finally:
        # Cleanup
        if os.path.exists(config.data.dataset_dir):
            logger.info(f"Cleaning up temporary dataset directory: {config.data.dataset_dir}")
            shutil.rmtree(config.data.dataset_dir)

if __name__ == '__main__':
    main()
