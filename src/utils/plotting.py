import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_training_curve(history_df: pd.DataFrame, output_dir: str):
    """
    Plots and saves the training and validation loss curve.

    Args:
        history_df (pd.DataFrame): DataFrame containing 'epoch', 'train_loss', 
                                   and 'val_loss' columns.
        output_dir (str): The directory to save the plot in.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history_df['epoch'], history_df['train_loss'], label='Training Loss')
    plt.plot(history_df['epoch'], history_df['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MAE)')
    plt.title('Training and Validation Loss Curve')
    plt.legend()
    plt.grid(True)
    curve_path = os.path.join(output_dir, 'training_curve.png')
    plt.savefig(curve_path)
    plt.close()
    print(f"Training curve saved to {curve_path}")

def plot_parity(targets: np.ndarray, preds: np.ndarray, output_dir: str):
    """
    Creates and saves a parity plot for the test set.

    Args:
        targets (np.ndarray): The ground truth values.
        preds (np.ndarray): The predicted values.
        output_dir (str): The directory to save the plot in.
    """
    plt.figure(figsize=(8, 8))
    plt.scatter(targets, preds, alpha=0.5, s=10) # Smaller points
    
    # Determine the limits for the plot
    lims = [
        min(min(targets), min(preds)),
        max(max(targets), max(preds)),
    ]
    
    # Add a y=x line
    plt.plot(lims, lims, 'r--', alpha=0.75, zorder=0, label='y=x')
    
    plt.xlabel('Actual BDE (kcal/mol)')
    plt.ylabel('Predicted BDE (kcal/mol)')
    plt.title('Parity Plot for Test Set')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.tight_layout()
    parity_plot_path = os.path.join(output_dir, 'parity_plot.png')
    plt.savefig(parity_plot_path)
    plt.close()
    print(f"Parity plot saved to {parity_plot_path}")
