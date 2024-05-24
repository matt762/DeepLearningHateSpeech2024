import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split
from datasets import load_dataset
import pandas as pd

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

def preprocess_data(dataset):
    features = ['sentiment', 'respect', 'insult', 'humiliate', 'status', 'dehumanize', 'violence', 'genocide', 'attack_defend']
    inputs = np.array([[entry[feature] for feature in features] for entry in dataset])
    labels = np.array([entry['hate_speech_score'] for entry in dataset])
    return inputs, labels

def load_best_hyperparameters(csv_file):
    df = pd.read_csv(csv_file)
    best_trial_idx = df['best_val_loss'].idxmin()
    best_hyperparams = df.loc[best_trial_idx]
    return best_hyperparams

def train_model_with_best_hyperparameters(csv_file):
    best_hyperparams = load_best_hyperparameters(csv_file)
    
    hidden_layers = int(best_hyperparams['hidden_layers'])
    hidden_units = int(best_hyperparams['hidden_units'])
    dropout_rate = float(best_hyperparams['dropout_rate'])
    learning_rate = float(best_hyperparams['learning_rate'])
    weight_decay = float(best_hyperparams['weight_decay'])
    batch_size = int(best_hyperparams['batch_size'])
    
    model = HateSpeechScoreModel(input_dim=9, hidden_layers=hidden_layers, hidden_units=hidden_units, dropout_rate=dropout_rate)
    criterion = nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    dataset = load_dataset('ucberkeley-dlab/measuring-hate-speech', split='train', cache_dir='./cache')
    inputs, labels = preprocess_data(dataset)

    X_train, X_val, y_train, y_val = train_test_split(inputs, labels, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_dataset = torch.utils.data.TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    device = torch.device("cpu")
    model.to(device)

    epochs = 200
    best_val_loss = float('inf')
    patience_counter = 0
    patience_limit = 40
    gradient_clip_value = 1.0

    print(f"Best hyperparameters:\nHidden Layers: {hidden_layers}\nHidden Units: {hidden_units}\nDropout Rate: {dropout_rate}\nLearning Rate: {learning_rate}\nWeight Decay: {weight_decay}\nBatch Size: {batch_size}")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets.view(-1, 1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_value)
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets.view(-1, 1))
                val_loss += loss.item() * inputs.size(0)

        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience_limit:
            break

        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    torch.save(model.state_dict(), 'hate_speech_neural_network.pth')
    print("Training complete.")

if __name__ == "__main__":
    csv_file = 'optuna_results.csv'

    train_model_with_best_hyperparameters(csv_file)