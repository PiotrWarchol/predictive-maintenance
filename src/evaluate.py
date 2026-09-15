import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_prep import prepare_data
from src.model import LSTMPredictor, get_device

def load_best_model(device):
    """Load the best saved model."""
    model = LSTMPredictor(
        input_size=14,
        hidden_size=128,
        num_layers=2,
        dropout=0.2
    )
    model.load_state_dict(
        torch.load('models/best_model.pth', map_location=device)
    )
    model = model.to(device)
    model.eval()
    print("Best model loaded successfully!")
    return model

def get_predictions(model, loader, device):
    """Run inference on test set."""
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for sequences, labels in loader:
            sequences = sequences.to(device)
            preds = model(sequences)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    return np.array(all_preds), np.array(all_labels)

def plot_predictions_vs_actual(preds, labels):
    """Plot predicted vs actual RUL for all test engines."""
    os.makedirs('outputs', exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Scatter plot — predicted vs actual
    axes[0].scatter(labels, preds, alpha=0.6, 
                   color='steelblue', edgecolors='white',
                   linewidth=0.5, s=60)
    
    # Perfect prediction line
    min_val = min(labels.min(), preds.min())
    max_val = max(labels.max(), preds.max())
    axes[0].plot([min_val, max_val], [min_val, max_val],
                'r--', linewidth=2, label='Perfect prediction')
    
    axes[0].set_xlabel('True RUL (cycles)', fontsize=12)
    axes[0].set_ylabel('Predicted RUL (cycles)', fontsize=12)
    axes[0].set_title('Predicted vs Actual RUL\nTest Engines',
                     fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # Error distribution
    errors = preds - labels
    axes[1].hist(errors, bins=20, color='steelblue',
                edgecolor='white', linewidth=0.5)
    axes[1].axvline(x=0, color='red', linestyle='--',
                   linewidth=2, label='Zero error')
    axes[1].axvline(x=errors.mean(), color='orange',
                   linestyle='--', linewidth=2,
                   label=f'Mean error: {errors.mean():.1f}')
    axes[1].set_xlabel('Prediction Error (cycles)', fontsize=12)
    axes[1].set_ylabel('Count', fontsize=12)
    axes[1].set_title('Error Distribution\nTest Engines',
                     fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('LSTM Predictive Maintenance — RUL Prediction Results',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/predictions_vs_actual.png',
                dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved outputs/predictions_vs_actual.png")

def plot_engine_predictions(preds, labels, n_engines=20):
    """
    Plot predicted vs actual RUL for individual engines.
    Shows how well the model tracks each engine's degradation.
    """
    fig, ax = plt.subplots(figsize=(14, 6))

    engine_ids = np.arange(1, n_engines + 1)
    x = np.arange(n_engines)
    width = 0.35

    bars1 = ax.bar(x - width/2, labels[:n_engines],
                  width, label='True RUL',
                  color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, preds[:n_engines],
                  width, label='Predicted RUL',
                  color='orange', alpha=0.8)

    ax.set_xlabel('Engine ID', fontsize=12)
    ax.set_ylabel('Remaining Useful Life (cycles)', fontsize=12)
    ax.set_title(f'True vs Predicted RUL — First {n_engines} Engines',
                fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(engine_ids, fontsize=9)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('outputs/engine_predictions.png',
                dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved outputs/engine_predictions.png")

def evaluate_model():
    """Run complete model evaluation."""
    print("Running model evaluation...")
    print("=" * 50)

    device = get_device()
    _, test_loader, _ = prepare_data()
    model = load_best_model(device)

    print("\nRunning inference on test engines...")
    preds, labels = get_predictions(model, test_loader, device)

    # Clip predictions to valid range
    preds = np.clip(preds, 0, 125)

    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(labels, preds))
    mae = mean_absolute_error(labels, preds)
    r2 = r2_score(labels, preds)
    errors = preds - labels
    mean_error = errors.mean()
    std_error = errors.std()

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Test engines:  {len(preds)}")
    print(f"RMSE:          {rmse:.2f} cycles")
    print(f"MAE:           {mae:.2f} cycles")
    print(f"R² Score:      {r2:.4f}")
    print(f"Mean Error:    {mean_error:.2f} cycles")
    print(f"Std Error:     {std_error:.2f} cycles")
    print()

    # Per-range analysis
    print("Performance by RUL range:")
    ranges = [(0, 30), (30, 60), (60, 90), (90, 125)]
    for low, high in ranges:
        mask = (labels >= low) & (labels < high)
        if mask.sum() > 0:
            range_rmse = np.sqrt(mean_squared_error(
                labels[mask], preds[mask]
            ))
            print(f"  RUL {low:3d}-{high:3d} cycles: "
                  f"RMSE={range_rmse:.2f} "
                  f"({mask.sum()} engines)")

    print()
    print("=" * 50)
    print("MAINTENANCE SCHEDULING CONTEXT")
    print("=" * 50)
    within_10 = (np.abs(errors) <= 10).sum()
    within_20 = (np.abs(errors) <= 20).sum()
    within_30 = (np.abs(errors) <= 30).sum()
    print(f"Predictions within ±10 cycles: "
          f"{within_10}/{len(preds)} "
          f"({100*within_10/len(preds):.1f}%)")
    print(f"Predictions within ±20 cycles: "
          f"{within_20}/{len(preds)} "
          f"({100*within_20/len(preds):.1f}%)")
    print(f"Predictions within ±30 cycles: "
          f"{within_30}/{len(preds)} "
          f"({100*within_30/len(preds):.1f}%)")

    if rmse < 15:
        print(f"\nExcellent result — RMSE of {rmse:.2f} cycles "
              f"is near state of the art for this dataset")
    elif rmse < 20:
        print(f"\nStrong result — RMSE of {rmse:.2f} cycles "
              f"is well suited for maintenance scheduling")
    else:
        print(f"\nGood result — RMSE of {rmse:.2f} cycles "
              f"provides useful maintenance windows")

    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_predictions_vs_actual(preds, labels)
    plot_engine_predictions(preds, labels)

    print("\nEvaluation complete!")
    print("All outputs saved to outputs/ folder")

    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'predictions': preds,
        'labels': labels
    }

if __name__ == "__main__":
    evaluate_model()