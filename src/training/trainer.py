import os
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import r2_score, mean_squared_error

from src.config import TrainConfig
from src.utils.reporting import save_training_log, save_test_metrics
from src.utils.plotting import plot_training_curve, plot_parity

class Trainer:
    """
    Handles the model training, validation, and evaluation pipeline.
    """
    def __init__(self, model, optimizer, train_loader, val_loader, test_loader, device, cfg: TrainConfig, run_dir: str):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.cfg = cfg
        self.run_dir = run_dir
        self.model_save_path = os.path.join(run_dir, cfg.model_save_path)

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
        Evaluates the best model on the test set and saves the results.
        Also generates a parity plot for train, val, and test sets.
        """
        print(f"\nLoading best model from {self.model_save_path} and evaluating on all sets...")
        self.model.load_state_dict(torch.load(self.model_save_path, map_location=self.device))
        self.model.eval()
        
        # Helper function to get predictions and targets for a given loader
        def _get_preds_targets(loader):
            y_preds = []
            y_trues = []
            with torch.no_grad():
                for batch in tqdm(loader, desc=f"[Predicting on {loader.dataset.root.split('/')[-1]} set]"):
                    batch = batch.to(self.device)
                    pred = self.model(batch)
                    if batch.mask.sum() > 0:
                        y_preds.append(pred[batch.mask].cpu().numpy())
                        y_trues.append(batch.y[batch.mask].cpu().numpy())
            if not y_preds:
                return np.array([]), np.array([])
            return np.concatenate(y_trues), np.concatenate(y_preds)

        # Get predictions and targets for each set
        y_true_train, y_pred_train = _get_preds_targets(self.train_loader)
        y_true_val, y_pred_val = _get_preds_targets(self.val_loader)
        y_true_test, y_pred_test = _get_preds_targets(self.test_loader)

        # Collect results for plotting
        results_for_plotting = {
            "train": (y_true_train, y_pred_train),
            "validation": (y_true_val, y_pred_val),
            "test": (y_true_test, y_pred_test),
        }

        # Save test metrics
        if y_true_test.size > 0:
            mae_test = mean_absolute_error(y_true_test, y_pred_test)
            mse_test = mean_squared_error(y_true_test, y_pred_test)
            rmse_test = np.sqrt(mse_test)
            r2_test = r2_score(y_true_test, y_pred_test)

            metrics = {'mae': mae_test, 'mse': mse_test, 'rmse': rmse_test, 'r2': r2_test}
            save_test_metrics(metrics, self.run_dir)
        else:
            print("No data in test set for metric calculation.")
            
        # Generate parity plot
        plot_parity(
            results=results_for_plotting,
            title="BDE Prediction Parity Plot",
            output_path=os.path.join(self.run_dir, "parity_plot.png")
        )


