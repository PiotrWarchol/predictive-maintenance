import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import os
import time

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_prep import prepare_data
from src.model import LSTMPredictor, get_device, count_parameters

def train_one_epoch(model, loader, optimizer, criterion, device):
    """Run one training epoch."""
    model.train()
    total_loss = 0

    for sequences, labels in loader:
        sequences = sequences.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        predictions = model(sequences)
        loss = criterion(predictions, labels)
        loss.backward()

        # Gradient clipping — prevents exploding gradients in LSTM
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)

def evaluate(model, loader, criterion, device):
    """Evaluate model and return loss and RMSE."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for sequences, labels in loader:
            sequences = sequences.to(device)
            labels = labels.to(device)

            predictions = model(sequences)
            loss = criterion(predictions, labels)
            total_loss += loss.item()

            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    rmse = np.sqrt(np.mean(
        (np.array(all_preds) - np.array(all_labels)) ** 2
    ))

    return avg_loss, rmse

def train(epochs=50):
    """Full training pipeline."""
    print("Starting predictive maintenance model training...")
    print("=" * 50)

    # Setup
    device = get_device()
    train_loader, test_loader, scaler = prepare_data()

    # Build model
    model = LSTMPredictor(
        input_size=14,
        hidden_size=128,
        num_layers=2,
        dropout=0.2
    )
    model = model.to(device)

    total, trainable = count_parameters(model)
    print(f"\nTrainable parameters: {trainable:,}")

    # Loss — MSE for regression
    criterion = nn.MSELoss()

    # Optimizer with weight decay for regularization
    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=1e-5
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=5,
        factor=0.5
    )

    # Training history
    history = {
        'train_loss': [],
        'test_loss': [],
        'test_rmse': []
    }

    best_rmse = float('inf')
    os.makedirs('models', exist_ok=True)

    print(f"\nTraining for {epochs} epochs...")
    print("-" * 50)

    for epoch in range(epochs):
        start = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        test_loss, test_rmse = evaluate(
            model, test_loader, criterion, device
        )

        scheduler.step(test_rmse)

        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['test_rmse'].append(test_rmse)

        elapsed = time.time() - start

        # Print every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Test Loss: {test_loss:.4f} | "
                  f"Test RMSE: {test_rmse:.2f} cycles | "
                  f"Time: {elapsed:.0f}s")

        # Save best model
        if test_rmse < best_rmse:
            best_rmse = test_rmse
            torch.save(model.state_dict(), 'models/best_model.pth')

    # Save history
    history_serializable = {
        key: [float(v) for v in values]
        for key, values in history.items()
    }
    with open('models/training_history.json', 'w') as f:
        json.dump(history_serializable, f)

    print("\n" + "=" * 50)
    print(f"Training complete!")
    print(f"Best Test RMSE: {best_rmse:.2f} cycles")
    print(f"Model saved to: models/best_model.pth")

    return model, history, best_rmse

if __name__ == "__main__":
    model, history, best_rmse = train(epochs=50)