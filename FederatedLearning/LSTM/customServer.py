from typing import List, Tuple, Dict, Any
from flwr.server import ServerApp, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.common import Metrics
import logging
import torch
import flwr as fl

logging.basicConfig(level=logging.INFO)

# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate metrics by computing weighted average of accuracy."""
    try:
        accuracies = [num_examples * m.get("accuracy", 0) for num_examples, m in metrics]
        examples = [num_examples for num_examples, _ in metrics]
        return {"accuracy": sum(accuracies) / sum(examples) if examples else 0}
    except Exception as e:
        logging.error(f'Error during metrics aggregation: {e}')
        return {"accuracy": 0}

class FedAvgWithModelSaving(FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client_id_mapping = {}
        self.next_client_id = 0
        self.previous_model_parameters = None

    def aggregate_fit(self, rnd: int, results: List[Tuple[Any, Dict[str, torch.Tensor]]], failures: List[Tuple[int, Exception]]) -> Dict[str, torch.Tensor]:
        try:
            # Aggregate the local models
            aggregated_parameters = super().aggregate_fit(rnd, results, failures)
            
            # Save the global model after aggregation
            model_file_name = f'modeloAgregado_r{rnd + 1}.pth'
            torch.save(aggregated_parameters, model_file_name)
            logging.info(f'Saved global model: {model_file_name}')
            
            # Save each local model
            for client, client_weights in results:
                original_client_id = client.cid  # Extract the original client id

                # Map original client id to a simpler id if not already mapped
                if original_client_id not in self.client_id_mapping:
                    self.client_id_mapping[original_client_id] = self.next_client_id
                    self.next_client_id += 1
                
                simple_client_id = self.client_id_mapping[original_client_id]
                logging.info(f'Client ID: {original_client_id} mapped to {simple_client_id}')

                client_model_file_name = f'submodelo_r{rnd + 1}_c{simple_client_id}.pth'
                torch.save(client_weights, client_model_file_name)
                logging.info(f'Saved local model for client {simple_client_id}: {client_model_file_name}')

            # Compare the weights
            self.compare_weights(rnd, aggregated_parameters)

            # Update the previous model parameters for the next round
            self.previous_model_parameters = aggregated_parameters

            return aggregated_parameters

        except Exception as e:
            logging.error(f'Error during aggregation or model saving: {e}')
            raise

    def compare_weights(self, rnd: int, curr_model_parameters: Dict[str, torch.Tensor]):
        """Compares the weights of the model between rounds."""
        if self.previous_model_parameters is None:
            logging.info('Not enough data to compare weights.')
            return

        try:
            # Compare parameters
            for param_name in curr_model_parameters.keys():
                prev_param = self.previous_model_parameters[param_name]
                curr_param = curr_model_parameters[param_name]
                if not torch.allclose(prev_param, curr_param):
                    logging.info(f'Weights have changed for parameter: {param_name}')
                else:
                    logging.info(f'Weights are unchanged for parameter: {param_name}')

        except Exception as e:
            logging.error(f'Error during weight comparison: {e}')

# Define metric aggregation function for evaluation
def aggregate_evaluate(rnd, results, failures):
    total_loss = 0.0
    total_accuracy = 0.0
    num_examples = 0

    for loss, num_examples, metrics in results:
        total_loss += loss * num_examples
        total_accuracy += metrics["accuracy"] * num_examples
        num_examples += num_examples

    avg_loss = total_loss / num_examples
    avg_accuracy = total_accuracy / num_examples

    return {"loss": avg_loss, "accuracy": avg_accuracy}

# Define strategy
strategy = FedAvgWithModelSaving(
    evaluate_metrics_aggregation_fn=weighted_average,
    fit_metrics_aggregation_fn=weighted_average  # Assuming you want the same aggregation function for fit metrics
)

# Define config
config = ServerConfig(num_rounds=10)  # Ensure this matches the number of rounds you intend to run

# Flower ServerApp
app = ServerApp(
    config=config,
    strategy=strategy,
)

# Legacy mode
if __name__ == "__main__":
    from flwr.server import start_server

    start_server(
        server_address="0.0.0.0:8081",
        config=config,
        strategy=strategy,
    )
