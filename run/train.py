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

def prepare_data(df: pd.DataFrame) -> List[Tuple[str, Dict[Tuple[int, int], float]]]:
    """
    Processes a DataFrame into a list of (SMILES, bde_labels_dict) tuples.
    """
    smiles_data = []
    
    # Group by molecule SMILES
    grouped = df.groupby('molecule')
    
    unique_smiles = list(grouped.groups.keys())

    print(f"Preparing data for {len(unique_smiles)} molecules...")
    for smiles in tqdm(unique_smiles):
        mol_df = grouped.get_group(smiles)
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        mol = Chem.AddHs(mol)  # Ensure hydrogens are added for consistent indexing

        bde_labels_dict = {}
        for _, row in mol_df.iterrows():
            bond_idx = int(row['bond_index'])
            bde = float(row['bde'])
            
            try:
                num_bonds = mol.GetNumBonds()
                if bond_idx >= num_bonds:
                    print(f"Warning: Skipping bond with index {bond_idx} for SMILES '{smiles}' because it is out of bounds (molecule has {num_bonds} bonds).")
                    continue

                bond = mol.GetBondWithIdx(bond_idx)
                u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                canonical_bond_key = (min(u, v), max(u, v))
                bde_labels_dict[canonical_bond_key] = bde
            except Exception as e:
                print(f"Warning: Skipping bond for SMILES '{smiles}' due to an unexpected error on bond_idx {bond_idx}: {e}")
                pass
                
        smiles_data.append((smiles, bde_labels_dict))
        
    return smiles_data

def main():
    parser = argparse.ArgumentParser(description="Train BDE Prediction Model")
    parser.add_argument('--data_path', type=str, default='examples/test_data.csv.gz', help='Path to the training data CSV')
    parser.add_argument('--vocab_path', type=str, default='etc/preprocessor.json', help='Path to the predefined vocabulary JSON')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size') # Increased default batch size
    parser.add_argument('--num_mols_overfit', type=int, default=-1, help='Number of molecules to use for overfitting test. Set to -1 to use all data.')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load Data
    print("Loading raw data...")
    df = pd.read_csv(args.data_path)
    
    # 2. Prepare and Split Data
    # Optionally sample a subset of molecules for overfitting tests
    if args.num_mols_overfit > 0:
        print(f"Sampling {args.num_mols_overfit} unique molecules for this run...")
        unique_smiles = df['molecule'].unique()
        if len(unique_smiles) < args.num_mols_overfit:
            print(f"Warning: Requested {args.num_mols_overfit} molecules, but only {len(unique_smiles)} unique molecules are available in the dataset.")
        else:
            # Use pandas sampling on the unique smiles to ensure randomness
            sampled_smiles = pd.Series(unique_smiles).sample(args.num_mols_overfit, random_state=RANDOM_SEED)
            df = df[df['molecule'].isin(sampled_smiles)]

    smiles_data = prepare_data(df)
    
    print("Splitting data into training, validation, and test sets...")
    # First split: 80% train+val, 20% test
    train_val_data, test_data = train_test_split(smiles_data, test_size=0.1, random_state=RANDOM_SEED)
    # Second split: 80% train, 10% val (relative to original size)
    train_data, val_data = train_test_split(train_val_data, test_size=0.1111, random_state=RANDOM_SEED) # 0.1111 * 0.9 = ~0.1

    print(f"Training set size: {len(train_data)}")
    print(f"Validation set size: {len(val_data)}")
    print(f"Test set size: {len(test_data)}")

    # 3. Initialize Tokenizer, Datasets, DataLoaders
    print("Initializing tokenizer and datasets...")
    if not os.path.exists(args.vocab_path):
        raise FileNotFoundError(f"Vocabulary file not found at {args.vocab_path}")
    tokenizer = Tokenizer(vocab_filepath=args.vocab_path)
    
    # Use separate directories for each dataset split
    train_dataset = BDEDataset(root='temp_dataset/train', smiles_data=train_data, tokenizer=tokenizer)
    val_dataset = BDEDataset(root='temp_dataset/val', smiles_data=val_data, tokenizer=tokenizer)
    test_dataset = BDEDataset(root='temp_dataset/test', smiles_data=test_data, tokenizer=tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
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
    print("Starting training...")
    best_val_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
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
            torch.save(model.state_dict(), 'best_model.pt')
            print(f"  -> New best validation loss: {best_val_loss:.4f}. Model saved to best_model.pt")

    print("Training finished.")
    
    # 6. Final Test Evaluation
    print("\nLoading best model and evaluating on test set...")
    model.load_state_dict(torch.load('best_model.pt'))
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

    # Clean up
    import shutil
    if os.path.exists('temp_dataset'):
        shutil.rmtree('temp_dataset')
    if os.path.exists('best_model.pt'):
        os.remove('best_model.pt')

if __name__ == '__main__':
    main()
