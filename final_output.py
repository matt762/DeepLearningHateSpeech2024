import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

# Load best hyperparameters from CSV
def load_best_hyperparameters(csv_file):
    df = pd.read_csv(csv_file)
    best_trial_idx = df['best_val_loss'].idxmin()
    best_hyperparams = df.loc[best_trial_idx]
    return best_hyperparams

best_hyperparams = load_best_hyperparameters('optuna_results.csv')

# Extract hyperparameters
hidden_layers = int(best_hyperparams['hidden_layers'])
hidden_units = int(best_hyperparams['hidden_units'])
dropout_rate = float(best_hyperparams['dropout_rate'])

class ResidualBlock(nn.Module):
    def __init__(self, input_dim, hidden_units, dropout_rate):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_units)
        self.bn1 = nn.BatchNorm1d(hidden_units)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_units, input_dim)
        self.bn2 = nn.BatchNorm1d(input_dim)
        
    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class HateSpeechScoreModel(nn.Module):
    def __init__(self, input_dim, hidden_layers, hidden_units, dropout_rate):
        super(HateSpeechScoreModel, self).__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_units))
        layers.append(nn.BatchNorm1d(hidden_units))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        for _ in range(hidden_layers):
            layers.append(ResidualBlock(hidden_units, hidden_units, dropout_rate))
        layers.append(nn.Linear(hidden_units, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

input_dim = 9

model = HateSpeechScoreModel(input_dim, hidden_layers, hidden_units, dropout_rate)
model.load_state_dict(torch.load('trained_hate_speech_model_best.pth', map_location=torch.device('cpu')))
model.eval()

scaler = StandardScaler()
scaler.mean_ = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
scaler.scale_ = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

def predict_hate_speech_score(input_data):
    with torch.no_grad():
        input_data = np.array(input_data).reshape(1, -1)
        input_data = scaler.transform(input_data)
        input_tensor = torch.tensor(input_data, dtype=torch.float32)
        prediction = model(input_tensor)
        return prediction.item()

def main():
    input_data = []
    labels = [
        'sentiment', 'respect', 'insult', 'humiliate',
        'status', 'dehumanize', 'violence', 'genocide', 'attack_defend'
    ]

    print("Insert input: value must be between 0 and 4:")
    for label in labels:
        while True:
            try:
                value = float(input(f'{label}: '))
                if 0 <= value <= 4:
                    input_data.append(value)
                    break
                else:
                    print("Invalid input: value must be between 0 and 4")
            except ValueError:
                print("Invalid input: value must be between 0 and 4")
    
    predicted_score = predict_hate_speech_score(input_data)
    print(f'Predicted Hate Speech Score: {predicted_score:.4f}')

if __name__ == "__main__":
    main()