import subprocess
import os
import json
import logging
import datetime
import numpy as np
import flwr as fl
import tensorflow as tf
from tensorflow import keras
from typing import List, Dict, Tuple, Any

logging.basicConfig(level=logging.INFO)

metrics_path = "modelos_guardados/metrics.json"
save_model_directory = "modelos_guardados"

def aggregate_fit_metrics(metrics: List[Tuple[float, Dict[str, float]]]) -> Dict[str, float]:
    losses = [m[1]["loss"] for m in metrics]
    accuracies = [m[1]["accuracy"] for m in metrics]

    avg_loss = np.mean(losses)
    avg_accuracy = np.mean(accuracies)

    logging.info(f"Average loss = {avg_loss:.4f}, Average accuracy = {avg_accuracy:.4f}")

    return {"loss": avg_loss, "accuracy": avg_accuracy}

def save_metrics(metrics_data: Dict[int, Dict[str, float]]):
    with open(metrics_path, "w") as file:
        json.dump(metrics_data, file)

def load_metrics() -> Dict[int, Dict[str, float]]:
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as file:
            return json.load(file)
    return {}

def save_global_model(weights: List[np.ndarray], round_number: int):
    model = build_model()
    model.set_weights(weights)
    os.makedirs(save_model_directory, exist_ok=True)
    round_dir = os.path.join(save_model_directory, f'round_{round_number}')
    os.makedirs(round_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_path = os.path.join(round_dir, f"global_model_{timestamp}.keras")
    model.save(model_save_path)
    logging.info(f'Saved global model for round {round_number}: {model_save_path}')

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

# Define la estrategia
class CustomFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.client_id_mapping = {}
        self.next_client_id = 0
        self.previous_model_parameters = None

    def aggregate_fit(self, rnd: int, results: List[Tuple[Any, fl.common.FitRes]], failures: List[BaseException]) -> Tuple[fl.common.Parameters, Dict[str, fl.common.Scalar]]:
        try:
            aggregated_parameters, aggregated_metrics = super().aggregate_fit(rnd, results, failures)
            
            if aggregated_parameters is not None:
                weights = fl.common.parameters_to_ndarrays(aggregated_parameters)
                save_global_model(weights, rnd)
            
            round_dir = os.path.join(save_model_directory, f'round_{rnd}')
            os.makedirs(round_dir, exist_ok=True)
            
            for client, fit_res in results:
                original_client_id = client.cid

                if original_client_id not in self.client_id_mapping:
                    self.client_id_mapping[original_client_id] = self.next_client_id
                    self.next_client_id += 1
                
                simple_client_id = self.client_id_mapping[original_client_id]
                logging.info(f'Client ID: {original_client_id} mapped to {simple_client_id}')

                client_model_file_name = os.path.join(round_dir, f'client_{simple_client_id}.keras')
                model = build_model()
                model.set_weights(fl.common.parameters_to_ndarrays(fit_res.parameters))
                model.save(client_model_file_name)
                logging.info(f'Saved local model for client {simple_client_id} for round {rnd}: {client_model_file_name}')

            self.previous_model_parameters = aggregated_parameters

            return aggregated_parameters, aggregated_metrics

        except Exception as e:
            logging.error(f'Error during aggregation or model saving: {e}')
            raise

strategy = CustomFedAvg(
    fraction_fit=1.0,
    min_fit_clients=2,
    min_available_clients=2,
    initial_parameters=None,
    on_fit_config_fn=lambda rnd: {"round": rnd},
    fit_metrics_aggregation_fn=aggregate_fit_metrics
)

def start_server():
    fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=6),
        strategy=strategy,
    )
    print_summary()

def print_summary():
    metrics_data = load_metrics()
    
    logging.info("\nTraining Summary:")
    for rnd, metrics in metrics_data.items():
        logging.info(f"Round {rnd}: Loss = {metrics['loss']:.4f}, Accuracy = {metrics['accuracy']:.4f}")

if __name__ == "__main__":
    start_server()
