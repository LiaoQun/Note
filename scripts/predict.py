"""
User-facing script to run BDE predictions on a list of SMILES strings.
"""
import argparse
import os
import pandas as pd

# Add project root to path to allow imports from src
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference.predictor import get_bde_predictions


def main():
    parser = argparse.ArgumentParser(description="Run BDE predictions using a trained model.")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--smiles', nargs='+', help='One or more SMILES strings to predict.')
    group.add_argument('--smiles_file', type=str, help='Path to a text file with one SMILES string per line.')

    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model checkpoint (.pt file).')
    parser.add_argument('--vocab_path', type=str, required=True, help='Path to the corresponding vocabulary file (.json).')
    parser.add_argument('--output_path', type=str, default='predictions.csv', help='Path to save the output CSV file.')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help="Device to run inference on.")
    parser.add_argument('--keep-duplicates', action='store_true', help="If set, keeps predictions for chemically equivalent bonds (e.g., in symmetric molecules).")
    
    args = parser.parse_args()

    # --- Load SMILES ---
    if args.smiles:
        smiles_list = args.smiles
    else:
        try:
            with open(args.smiles_file, 'r') as f:
                smiles_list = [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            print(f"Error: Smiles file not found at {args.smiles_file}")
            return
    
    if not smiles_list:
        print("Error: No SMILES strings to process.")
        return

    # --- Run Prediction using the new simple API ---
    print("Making predictions...")
    results_df = get_bde_predictions(
        smiles=smiles_list,
        model_path=args.model_path,
        vocab_path=args.vocab_path,
        drop_duplicates=not args.keep_duplicates,
        device=args.device
    )
    
    # --- Save and Display Results ---
    if results_df.empty:
        print("No valid bonds found for prediction in the provided molecules.")
    else:
        results_df.to_csv(args.output_path, index=False)
        print(f"\nPredictions successfully saved to {args.output_path}")
        print("\nPrediction results head:")
        print(results_df.head().to_string())
