import flwr as fl
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

# Definir la clase del modelo LSTM
class LSTMNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMNetwork, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out

def load_data():
    df = pd.read_csv('processed_data.csv')
    X = df.drop(columns=['diabetesMed_Yes']).values
    y = df['diabetesMed_Yes'].values
    X = X.reshape(X.shape[0], 1, X.shape[1])
    y = y.reshape(-1, 1)  # Ajustar las dimensiones de y
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    return DataLoader(dataset, batch_size=32, shuffle=True)
    
# Definir funciones de entrenamiento y evaluación
def train(net, trainloader, optimizer, criterion):
    net.train()
    for sequences, targets in trainloader:
        optimizer.zero_grad()
        outputs = net(sequences)
        loss = criterion(outputs.squeeze(), targets.squeeze())
        loss.backward()
        optimizer.step()

def evaluate(net, testloader, criterion):
    net.eval()
    loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for sequences, targets in testloader:
            outputs = net(sequences)
            loss += criterion(outputs.squeeze(), targets.squeeze()).item()
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
    accuracy = correct / total
    return loss / len(testloader), accuracy


# Crear cliente Flower
class FlowerClient(fl.client.NumPyClient):
    def __init__(self):
        super().__init__()
        self.net = LSTMNetwork(input_size=213, hidden_size=50, output_size=1)
        self.trainloader = load_data()
        self.testloader = load_data()
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.001)
        self.criterion = nn.BCEWithLogitsLoss()

    def get_parameters(self):
        return [val.cpu().numpy() for val in self.net.state_dict().values()]

    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(self.net, self.trainloader, self.optimizer, self.criterion)
        return self.get_parameters(), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        return evaluate(self.net, self.testloader, self.criterion)

# Iniciar el cliente Flower
if __name__ == "__main__":
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=FlowerClient())
