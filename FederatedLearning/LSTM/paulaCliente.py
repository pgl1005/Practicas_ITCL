import flwr as fl
import tensorflow as tf
from tensorflow import keras
import numpy as np
from typing import Tuple, Dict, List
import os
import datetime
import logging

logging.basicConfig(level=logging.INFO)

def load_data() -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    return (x_train, y_train), (x_test, y_test)

def build_model() -> keras.Model:
    model = keras.Sequential([
        keras.Input(shape=(28, 28)),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

(x_train, y_train), (x_test, y_test) = load_data()

model = build_model()

class MnistClient(fl.client.NumPyClient):
    def get_parameters(self, config: Dict[str, str]) -> List[np.ndarray]:
        return model.get_weights()

    def fit(self, parameters: List[np.ndarray], config: Dict[str, str]) -> Tuple[List[np.ndarray], int, Dict]:
        model.set_weights(parameters)
        history = model.fit(x_train, y_train, epochs=1, batch_size=32, verbose=2)
        
        save_dir = "modelos_guardados"
        os.makedirs(save_dir, exist_ok=True)
        
        try:
       
            model.save(model_save_path)
            logging.info(f'Saved model to {model_save_path}')
        except Exception as e:
            logging.error(f"Error saving model: {e}")  
        
        return model.get_weights(), len(x_train), {
            "loss": history.history["loss"][-1],
            "accuracy": history.history["accuracy"][-1],
        }

    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, str]) -> Tuple[float, int, Dict[str, float]]:
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
        return loss, len(x_test), {"accuracy": accuracy}

fl.client.start_client(
    server_address="localhost:8080",
    client=MnistClient().to_client()  
)
