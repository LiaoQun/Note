import os
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import r2_score, mean_squared_error

from src.config import TrainConfig
from src.utils.reporting import save_training_log
from src.utils.plotting import plot_training_curve, plot_parity
from src.inference.predictor import Predictor

class Trainer:
    """
    Handles the model training, validation, and evaluation pipeline.
    """
    def __init__(self, model, optimizer, train_loader, val_loader, test_loader, 
                 device, cfg: TrainConfig, run_dir: str, full_dataset_df: pd.DataFrame, 
                 data_splits: Dict[str, List], vocab_path: str):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.cfg = cfg
        self.run_dir = run_dir
        self.model_save_path = os.path.join(run_dir, cfg.model_save_path)
        self.vocab_path = vocab_path
        self.full_dataset_df = full_dataset_df
        self.data_splits = data_splits

    def train(self):
        """
        Executes the main training loop, including validation and early stopping.
        """
        print("Starting training...")
        best_val_loss = float('inf')
        patience_counter = 0
        history = []

        for epoch in range(1, self.cfg.epochs + 1):
            # Train one epoch
            avg_train_loss = self._train_epoch(epoch)
            
            # Validate one epoch
            avg_val_loss = self._validate_epoch(epoch)
            
            print(f"Epoch {epoch:03d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
            
            history.append({
                'epoch': epoch,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss
            })

            # Early stopping and model saving
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(self.model.state_dict(), self.model_save_path)
                print(f"  -> New best validation loss: {best_val_loss:.4f}. Model saved to {self.model_save_path}")
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"  -> Validation loss did not improve. Patience: {patience_counter}/{self.cfg.early_stopping_patience}")

            if patience_counter >= self.cfg.early_stopping_patience:
                print("\nEarly stopping triggered.")
                break
        
        print("\nTraining finished.")
        
        # Save logs and plots
        history_df = save_training_log(history, self.run_dir)
        plot_training_curve(history_df, self.run_dir)

    def _train_epoch(self, epoch: int) -> float:
        """Handles the training logic for a single epoch."""
        self.model.train()
        total_loss = 0
        for batch in tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]", leave=False):
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            pred = self.model(batch)
            if batch.mask.sum() > 0:
                loss = F.l1_loss(pred[batch.mask], batch.y[batch.mask])
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item() * batch.num_graphs
        return total_loss / len(self.train_loader.dataset) if len(self.train_loader.dataset) > 0 else 0

    def _validate_epoch(self, epoch: int) -> float:
        """Handles the validation logic for a single epoch."""
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]", leave=False):
                batch = batch.to(self.device)
                pred = self.model(batch)
                if batch.mask.sum() > 0:
                    loss = F.l1_loss(pred[batch.mask], batch.y[batch.mask])
                    total_loss += loss.item() * batch.num_graphs
        return total_loss / len(self.val_loader.dataset) if len(self.val_loader.dataset) > 0 else 0

    def evaluate(self):
        """
        Evaluates the best model on the test set and saves the results.
        """
        print(f"\nLoading best model from {self.model_save_path} and evaluating on test set...")
        self.model.load_state_dict(torch.load(self.model_save_path))
        self.model.eval()
        
    def evaluate(self):
        """
        Evaluates the best model on all data splits (train, val, test),
        saves the full predictions, and generates plots.
        """
        print(f"\nLoading best model from {self.model_save_path} for final evaluation...")
        
        try:
            # 1. Initialize predictor with the best model
            predictor = Predictor(
                model_path=self.model_save_path,
                vocab_path=self.vocab_path,
                device=self.device
            )
        except FileNotFoundError as e:
            print(f"Could not initialize predictor: {e}. Aborting evaluation.")
            return

        results_for_plotting = {}
        
        # 2. Iterate through each data split, make predictions, and save results
        for split_name, data_list in self.data_splits.items():
            print(f"\n--- Predicting on {split_name} set ---")
            if not data_list:
                print(f"{split_name} set is empty. Skipping.")
                continue

            # Get unique SMILES for the current split
            smiles_list = sorted(list(set([item[0] for item in data_list])))
            
            # Get rich prediction dataframe using the predictor
            pred_df = predictor.predict(smiles_list, drop_duplicates=False)

            # Merge with original dataframe to get ground truth `bde`
            # The original df has all ground truth data
            merged_df = pd.merge(
                pred_df,
                self.full_dataset_df[['molecule', 'bond_index', 'bde']],
                on=['molecule', 'bond_index'],
                how='inner' # Use inner merge to only keep bonds with known ground truth
            )

            # Save the detailed predictions to a CSV file
            output_path = os.path.join(self.run_dir, f'predictions_{split_name}.csv')
            merged_df.to_csv(output_path, index=False)
            print(f"Saved detailed predictions for {split_name} set to {output_path}")

            # Prepare data for parity plot
            if not merged_df.empty:
                y_true = merged_df['bde'].values
                y_pred = merged_df['bde_pred'].values
                results_for_plotting[split_name] = (y_true, y_pred)
        
        # 3. Generate parity plot
        if results_for_plotting:
            print("\nGenerating parity plot...")
            plot_parity(
                results=results_for_plotting,
                title="BDE Prediction Parity Plot",
                output_path=os.path.join(self.run_dir, "parity_plot.png")
            )
        else:
            print("No results to plot.")


