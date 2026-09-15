import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader
import os

# ─────────────────────────────────────────
# Dataset configuration
# ─────────────────────────────────────────
DATA_PATH = 'data/raw/CMaps'

COLUMNS = [
    'unit_id', 'cycle',
    'setting1', 'setting2', 'setting3',
    's1', 's2', 's3', 's4', 's5',
    's6', 's7', 's8', 's9', 's10',
    's11', 's12', 's13', 's14', 's15',
    's16', 's17', 's18', 's19', 's20', 's21'
]

# Sensors with meaningful signal for FD001
# Constant or near-constant sensors are dropped
USEFUL_SENSORS = [
    's2', 's3', 's4', 's7', 's8', 's9',
    's11', 's12', 's13', 's14', 's15',
    's17', 's20', 's21'
]

SEQUENCE_LENGTH = 30  # Use last 30 cycles to predict RUL
MAX_RUL = 125         # Cap RUL at 125 — piecewise linear model
BATCH_SIZE = 64

# ─────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────
def load_data(subset='FD001'):
    """Load train, test, and RUL files for a given subset."""
    train_df = pd.read_csv(
        f'{DATA_PATH}/train_{subset}.txt',
        sep=' ', header=None,
        names=COLUMNS, index_col=False
    ).dropna(axis=1)

    test_df = pd.read_csv(
        f'{DATA_PATH}/test_{subset}.txt',
        sep=' ', header=None,
        names=COLUMNS, index_col=False
    ).dropna(axis=1)

    rul_df = pd.read_csv(
        f'{DATA_PATH}/RUL_{subset}.txt',
        header=None, names=['RUL']
    )

    print(f"Loaded {subset}:")
    print(f"  Train engines: {train_df['unit_id'].nunique()}")
    print(f"  Train rows:    {len(train_df):,}")
    print(f"  Test engines:  {test_df['unit_id'].nunique()}")
    print(f"  Test rows:     {len(test_df):,}")

    return train_df, test_df, rul_df

# ─────────────────────────────────────────
# Feature engineering
# ─────────────────────────────────────────
def add_rul_to_train(df):
    """
    Add Remaining Useful Life column to training data.
    RUL = max_cycle - current_cycle for each engine.
    """
    max_cycles = df.groupby('unit_id')['cycle'].max().reset_index()
    max_cycles.columns = ['unit_id', 'max_cycle']
    df = df.merge(max_cycles, on='unit_id')
    df['RUL'] = df['max_cycle'] - df['cycle']
    df = df.drop('max_cycle', axis=1)
    return df

def clip_rul(df, max_rul=MAX_RUL):
    """
    Cap RUL at max_rul — piecewise linear degradation model.
    
    Engines don't degrade meaningfully until they approach
    failure. Capping at 125 focuses the model on the
    degradation phase that actually matters for maintenance.
    """
    df['RUL'] = df['RUL'].clip(upper=max_rul)
    return df

def normalize_sensors(train_df, test_df):
    """
    Normalize sensor readings using MinMaxScaler.
    Fit on training data only — transform both train and test.
    """
    scaler = MinMaxScaler()
    train_df[USEFUL_SENSORS] = scaler.fit_transform(
        train_df[USEFUL_SENSORS]
    )
    test_df[USEFUL_SENSORS] = scaler.transform(
        test_df[USEFUL_SENSORS]
    )
    return train_df, test_df, scaler

# ─────────────────────────────────────────
# Sequence creation
# ─────────────────────────────────────────
def create_train_sequences(df, sequence_length=SEQUENCE_LENGTH):
    """
    Create sliding window sequences from training data.
    
    For each engine, slide a window of sequence_length cycles
    across all readings. Each window predicts the RUL at the
    end of the window.
    """
    sequences = []
    labels = []

    for unit_id in df['unit_id'].unique():
        unit_data = df[df['unit_id'] == unit_id].sort_values('cycle')
        sensor_data = unit_data[USEFUL_SENSORS].values
        rul_data = unit_data['RUL'].values

        for i in range(len(sensor_data) - sequence_length + 1):
            sequences.append(sensor_data[i:i + sequence_length])
            labels.append(rul_data[i + sequence_length - 1])

    return np.array(sequences, dtype=np.float32), \
           np.array(labels, dtype=np.float32)

def create_test_sequences(test_df, rul_df,
                          sequence_length=SEQUENCE_LENGTH):
    """
    Create test sequences — one per engine using last
    sequence_length readings.
    """
    sequences = []

    for unit_id in test_df['unit_id'].unique():
        unit_data = test_df[
            test_df['unit_id'] == unit_id
        ].sort_values('cycle')
        sensor_data = unit_data[USEFUL_SENSORS].values

        if len(sensor_data) >= sequence_length:
            sequences.append(sensor_data[-sequence_length:])
        else:
            # Pad with zeros if not enough cycles
            pad_length = sequence_length - len(sensor_data)
            pad = np.zeros((pad_length, len(USEFUL_SENSORS)))
            sequences.append(np.vstack([pad, sensor_data]))

    # Clip test RUL at MAX_RUL
    test_rul = np.clip(
        rul_df['RUL'].values, 0, MAX_RUL
    ).astype(np.float32)

    return np.array(sequences, dtype=np.float32), test_rul

# ─────────────────────────────────────────
# PyTorch Dataset
# ─────────────────────────────────────────
class RULDataset(Dataset):
    """PyTorch Dataset for RUL prediction."""

    def __init__(self, sequences, labels):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

# ─────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────
def prepare_data(subset='FD001'):
    """
    Full data preparation pipeline.
    Returns train and test DataLoaders.
    """
    # Load raw data
    train_df, test_df, rul_df = load_data(subset)

    # Add RUL to training data
    train_df = add_rul_to_train(train_df)
    train_df = clip_rul(train_df)

    # Normalize sensors
    train_df, test_df, scaler = normalize_sensors(
        train_df, test_df
    )

    # Create sequences
    print("\nCreating sequences...")
    X_train, y_train = create_train_sequences(train_df)
    X_test, y_test = create_test_sequences(test_df, rul_df)

    print(f"Training sequences: {X_train.shape}")
    print(f"Training labels:    {y_train.shape}")
    print(f"Test sequences:     {X_test.shape}")
    print(f"Test labels:        {y_test.shape}")
    print(f"\nRUL statistics:")
    print(f"  Train RUL min: {y_train.min():.0f} cycles")
    print(f"  Train RUL max: {y_train.max():.0f} cycles")
    print(f"  Train RUL mean: {y_train.mean():.0f} cycles")

    # Create DataLoaders
    train_dataset = RULDataset(X_train, y_train)
    test_dataset = RULDataset(X_test, y_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    print(f"\nTraining batches: {len(train_loader)}")
    print(f"Test batches:     {len(test_loader)}")

    return train_loader, test_loader, scaler

if __name__ == "__main__":
    print("Running data preparation pipeline...")
    print("=" * 50)
    train_loader, test_loader, scaler = prepare_data()
    
    # Verify one batch
    sequences, labels = next(iter(train_loader))
    print(f"\nSample batch:")
    print(f"  Sequence shape: {sequences.shape}")
    print(f"  Label shape:    {labels.shape}")
    print(f"  Sequence dtype: {sequences.dtype}")
    print(f"  Label range:    [{labels.min():.0f}, "
          f"{labels.max():.0f}]")
    print("\nData preparation complete!")