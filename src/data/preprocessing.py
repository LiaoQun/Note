import os
import pandas as pd
from typing import List, Tuple, Dict
from rdkit import Chem
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

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
