import warnings
from collections import OrderedDict
import flwr as fl
from flwr.client import NumPyClient, start_client
import torch
from torch.optim import Adam
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd
from argparse import ArgumentParser
import logging
import os

logging.basicConfig(level=logging.INFO)

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Argument parser para permitir valores por defecto
parser = ArgumentParser(description="Flower Client")
parser.add_argument('--server_address', type=str, default="127.0.0.1:8080", help="Address of the Flower server")
parser.add_argument('--partition_id', type=int, default=0, help="Partition ID for the client")
parser.add_argument('--round', type=int, default=1, help="Current round number")  # Added round argument
args = parser.parse_args()

# Load and preprocess data
df = pd.read_csv('processed_data.csv')
X = df.drop(columns=['diabetesMed_Yes']).values
y = df['diabetesMed_Yes'].values

# Asegúrate de que las etiquetas son 0 o 1
y = (y > 0).astype(int)

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data into partitions
num_clients = 10
partition_size = len(X) // num_clients
partition_start = args.partition_id * partition_size
partition_end = partition_start + partition_size if args.partition_id != num_clients - 1 else len(X)

X_train, X_test, y_train, y_test = train_test_split(X[partition_start:partition_end], y[partition_start:partition_end], test_size=0.2, random_state=42)

class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)  # Ensure labels are Long type
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

trainset = CustomDataset(X_train, y_train)
testset = CustomDataset(X_test, y_test)

trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
testloader = DataLoader(testset, batch_size=32, shuffle=False)

class LSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.5)  # Add dropout for regularization
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out)
        out = self.fc(out[:, -1, :])
        return out

input_dim = X.shape[1]
hidden_dim = 2  
output_dim = 2  # Cambiado a 2 para clasificación binaria con CrossEntropyLoss
num_layers = 2
net = LSTMNet(input_dim, hidden_dim, num_layers, output_dim).to(DEVICE)

def train(model, dataloader, epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # L2 regularization
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
    model.train()
    
    for epoch in range(epochs):
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.unsqueeze(1)  # Ensure X_batch is 3D
            y_batch = y_batch.long()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        val_loss, _, _, _ = test(model, dataloader)  # Use a validation set instead of test set for early stopping
        scheduler.step(val_loss)

def test(model, dataloader):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    loss = 0
    correct = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.unsqueeze(1)  # Ensure X_batch is 3D
            y_batch = y_batch.long()
            outputs = model(X_batch)
            loss += criterion(outputs, y_batch).item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y_batch).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    
    accuracy = correct / len(dataloader.dataset)
    loss /= len(dataloader)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    
    print(f"Client num: {args.partition_id}")
    print(f"Test Metrics - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}, Loss: {loss:.4f}")
    
    return loss, accuracy, all_preds, all_labels

class FlowerClient(NumPyClient):
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(net, trainloader, epochs=1)
        
        # Crear carpeta para el modelo si no existe
        model_dir = f'models/round_{args.round}'
        os.makedirs(model_dir, exist_ok=True)
        
        # Guardar el modelo local después del entrenamiento
        model_path = os.path.join(model_dir, f'submodelo_c{args.partition_id + 1}.pth')
        torch.save(net.state_dict(), model_path)
        print(f'Modelo guardado en {model_path}')

        return self.get_parameters(config={}), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy, all_preds, all_labels = test(net, testloader)
        return loss, len(testloader.dataset), {"accuracy": accuracy}

def client_fn(cid: str):
    return FlowerClient().to_client()

if __name__ == "__main__":
    start_client(server_address=args.server_address, client=client_fn(""))







