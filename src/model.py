import torch
import torch.nn as nn

class LSTMPredictor(nn.Module):
    """
    LSTM model for Remaining Useful Life prediction.
    
    Takes a sequence of sensor readings over time and
    predicts how many cycles remain before maintenance
    is required.
    
    Architecture:
        Input: (batch, sequence_length, n_sensors)
        LSTM layers: extract temporal patterns
        Fully connected: map to RUL prediction
        Output: (batch, 1) — predicted RUL in cycles
    """
    
    def __init__(self,
                 input_size=14,
                 hidden_size=128,
                 num_layers=2,
                 dropout=0.2):
        super(LSTMPredictor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected classifier head
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, sequence_length, input_size)
        
        Returns:
            Predicted RUL (batch, 1)
        """
        # LSTM forward pass
        # out shape: (batch, sequence_length, hidden_size)
        out, _ = self.lstm(x)
        
        # Take only the last time step output
        # Shape: (batch, hidden_size)
        out = out[:, -1, :]
        
        # Fully connected layers
        out = self.fc(out)
        
        return out.squeeze(1)

def get_device():
    """Get best available device."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device

def count_parameters(model):
    """Count trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )
    return total, trainable

if __name__ == "__main__":
    print("Building LSTM predictive maintenance model...")
    print("=" * 50)
    
    device = get_device()
    
    model = LSTMPredictor(
        input_size=14,
        hidden_size=128,
        num_layers=2,
        dropout=0.2
    )
    model = model.to(device)
    
    total, trainable = count_parameters(model)
    print(f"\nModel: LSTM with {total:,} parameters")
    print(f"All trainable: {trainable:,}")
    print(f"\nArchitecture:")
    print(model)
    
    # Test forward pass
    dummy_input = torch.randn(64, 30, 14).to(device)
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"\nInput shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Sample predictions: {output[:5].tolist()}")
    print("\nModel built successfully!")