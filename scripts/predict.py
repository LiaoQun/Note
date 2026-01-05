"""
User-facing script to run BDE predictions on a list of SMILES strings.
"""
import argparse
import os
import json
import pandas as pd

# Add project root to path to allow imports from src
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.inference.predictor import get_bde_predictions


def main():
    parser = argparse.ArgumentParser(
        description="Run BDE predictions using a trained model. "
                    "Provide either a run directory OR the model and vocab paths directly."
    )
    
    # --- Input SMILES ---
    smiles_group = parser.add_mutually_exclusive_group(required=True)
    smiles_group.add_argument('--smiles', nargs='+', help='One or more SMILES strings to predict.')
    smiles_group.add_argument('--smiles_file', type=str, help='Path to a text file with one SMILES string per line.')

    # --- Model Loading ---
    model_group = parser.add_argument_group('Model Loading')
    model_group.add_argument('--run_dir', type=str, help='Path to the training run directory (e.g., training_runs/TIMESTAMP).')
    model_group.add_argument('--model_path', type=str, help='Path to the trained model checkpoint (.pt file).')
    model_group.add_argument('--vocab_path', type=str, help='Path to the corresponding vocabulary file (.json).')

    # --- Other Options ---
    parser.add_argument('--output_path', type=str, default='predictions.csv', help='Path to save the output CSV file.')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help="Device to run inference on.")
    parser.add_argument('--keep-duplicates', action='store_true', help="If set, keeps predictions for chemically equivalent bonds (e.g., in symmetric molecules).")
    
    args = parser.parse_args()

    # --- Determine Model and Vocab Paths ---
    model_path = args.model_path
    vocab_path = args.vocab_path

    if args.run_dir:
        if args.model_path or args.vocab_path:
            parser.error("Cannot specify --run_dir with --model_path or --vocab_path.")
        
        config_path = os.path.join(args.run_dir, 'config.json')
        if not os.path.exists(config_path):
            print(f"Error: config.json not found in the specified run directory: {args.run_dir}")
            return
            
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Construct model path from config
        model_filename = config.get('train', {}).get('model_save_path', 'bde_model.pt')
        model_path = os.path.join(args.run_dir, model_filename)

        # Determine vocab path: prefer run-specific, fallback to config
        run_vocab_path = os.path.join(args.run_dir, 'vocab.json')
        if os.path.exists(run_vocab_path):
            vocab_path = run_vocab_path
        else:
            vocab_path = config.get('data', {}).get('vocab_path', 'etc/preprocessor.json')
            
        print(f"Loading from run directory '{args.run_dir}':")
        print(f"  -> Model Path: {model_path}")
        print(f"  -> Vocab Path: {vocab_path}")

    elif not (model_path and vocab_path):
        parser.error("If not using --run_dir, both --model_path and --vocab_path are required.")


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
    print("\nMaking predictions...")
    results_df = get_bde_predictions(
        smiles=smiles_list,
        model_path=model_path,
        vocab_path=vocab_path,
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

if __name__ == '__main__':
    main()