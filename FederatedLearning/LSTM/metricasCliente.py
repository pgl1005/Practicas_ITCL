import flwr as fl
import tensorflow as tf
from tensorflow import keras
import numpy as np

# Definir el modelo de ejemplo
def create_model(input_shape):
    model = keras.Sequential([
        keras.layers.Input(shape=input_shape),
        keras.layers.LSTM(128, return_sequences=True),
        keras.layers.LSTM(128),
        keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

class Client(fl.client.NumPyClient):
    def __init__(self, model, data):
        self.model = model
        self.data = data

    def get_parameters(self):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(self.data['x_train'], self.data['y_train'], epochs=1, batch_size=32, verbose=0)
        return self.model.get_weights(), len(self.data['x_train']), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.data['x_test'], self.data['y_test'], verbose=0)
        return loss, len(self.data['x_test']), {"accuracy": accuracy}

# Definir la función start_client para aceptar argumentos
def start_client(server_address: str, client: fl.client.Client):
    fl.client.start_numpy_client(server_address=server_address, client=client)

# Cargar datos de ejemplo
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
data = {
    "x_train": x_train,
    "y_train": y_train,
    "x_test": x_test,
    "y_test": y_test
}

# Crear el modelo
model = create_model(input_shape=(28, 28))

# Crear el cliente FL
client = Client(model, data)

# Iniciar el cliente FL
if __name__ == "__main__":
    start_client("0.0.0.0:8081", client)
