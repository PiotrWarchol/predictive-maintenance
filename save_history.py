# Run this once to create the history file
import json

history = {
    'train_loss': [2938.56, 287.02, 252.87, 217.59, 193.15, 
                   171.91, 160.84, 155.30, 146.28, 145.39, 139.86],
    'test_loss': [1736.94, 201.59, 245.37, 186.64, 214.88,
                  295.65, 276.02, 249.64, 252.56, 247.86, 250.35],
    'test_rmse': [41.22, 13.73, 15.40, 12.99, 13.98,
                  16.57, 15.97, 15.13, 15.18, 15.03, 15.11]
}

with open('models/training_history.json', 'w') as f:
    json.dump(history, f)

print("Training history saved!")