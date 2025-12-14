import argparse
import os
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from rdkit import Chem
from tqdm import tqdm
from typing import List, Tuple, Dict
from sklearn.model_selection import train_test_split

# Adjust path to import from src
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.features.featurizer import Tokenizer
from src.data.dataset import BDEDataset
from src.models.mpnn import BDEModel

# Define constants from the original implementation
ATOM_FEATURES = 128
NUM_MESSAGES = 6
RANDOM_SEED = 42

def prepare_data(df: pd.DataFrame, num_mols: int = -1) -> List[Tuple[str, Dict[Tuple[int, int], float]]]:
    """
    Processes a DataFrame into a list of (SMILES, bde_labels_dict) tuples.
    """
    smiles_data = []
    
    # Group by molecule SMILES
    grouped = df.groupby('molecule')
    
    unique_smiles = list(grouped.groups.keys())
    if num_mols > 0:
        unique_smiles = unique_smiles[:num_mols]

    print(f"Preparing data for {len(unique_smiles)} molecules...")
    for smiles in tqdm(unique_smiles):
        mol_df = grouped.get_group(smiles)
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        
        bde_labels_dict = {}
        for _, row in mol_df.iterrows():
            bond_idx = int(row['bond_index'])
            bde = float(row['bde'])
            
            try:
                bond = mol.GetBondWithIdx(bond_idx)
                u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                canonical_bond_key = (min(u, v), max(u, v))
                bde_labels_dict[canonical_bond_key] = bde
            except Exception:
                pass
                
        smiles_data.append((smiles, bde_labels_dict))
        
    return smiles_data

def main():
    parser = argparse.ArgumentParser(description="Train BDE Prediction Model")
    parser.add_argument('--data_path', type=str, default='examples/test_data.csv.gz', help='Path to the training data CSV')
    parser.add_argument('--vocab_path', type=str, default='etc/preprocessor.json', help='Path to the predefined vocabulary JSON')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_mols_overfit', type=int, default=20, help='Number of molecules to use for overfitting test. Set to -1 to use all data.')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load Data
    print("Loading raw data...")
    df = pd.read_csv(args.data_path)
    
    # 2. Prepare and Split Data
    smiles_data = prepare_data(df, num_mols=args.num_mols_overfit)
    
    print("Splitting data into training and validation sets...")
    train_data, val_data = train_test_split(smiles_data, test_size=0.2, random_state=RANDOM_SEED)
    print(f"Training set size: {len(train_data)}, Validation set size: {len(val_data)}")

    # 3. Initialize Tokenizer, Datasets, DataLoaders
    print("Initializing tokenizer and datasets...")
    if not os.path.exists(args.vocab_path):
        raise FileNotFoundError(f"Vocabulary file not found at {args.vocab_path}")
    tokenizer = Tokenizer(vocab_filepath=args.vocab_path)
    
    train_dataset_root = 'temp_dataset/train'
    val_dataset_root = 'temp_dataset/val'
    train_dataset = BDEDataset(root=train_dataset_root, smiles_data=train_data, tokenizer=tokenizer)
    val_dataset = BDEDataset(root=val_dataset_root, smiles_data=val_data, tokenizer=tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 4. Initialize Model and Optimizer
    print("Initializing model...")
    model = BDEModel(
        num_atom_classes=tokenizer.atom_num_classes + 1,
        num_bond_classes=tokenizer.bond_num_classes + 1,
        atom_features=ATOM_FEATURES,
        num_messages=NUM_MESSAGES
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 5. Training Loop
    print("Starting training loop...")
    for epoch in range(1, args.epochs + 1):
        # Training
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

    print("Training finished.")
    # Clean up processed dataset folders
    import shutil
    if os.path.exists('temp_dataset'):
        shutil.rmtree('temp_dataset')

if __name__ == '__main__':
    main()
