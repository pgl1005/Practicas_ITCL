import flwr as fl

def weighted_average(metrics):
    if not metrics or isinstance(metrics, (int, float)):
        return {"accuracy": 0.0, "loss": 0.0}
    
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    losses = [num_examples * m.get("loss", 0.0) for num_examples, m in metrics]  # Use m.get("loss", 0.0) to handle missing key
    examples = [num_examples for num_examples, _ in metrics]
    
    if sum(examples) == 0:
        return {"accuracy": 0.0, "loss": 0.0}
    
    weighted_accuracy = sum(accuracies) / sum(examples)
    weighted_loss = sum(losses) / sum(examples)
    
    return {"accuracy": weighted_accuracy, "loss": weighted_loss}

strategy = fl.server.strategy.FedAvg(
    evaluate_metrics_aggregation_fn=weighted_average,
)

if __name__ == "__main__":
    hist = fl.server.start_server(
        server_address="127.0.0.1:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )

    # Imprimir las métricas finales
    final_metrics = hist.metrics_centralized
    print("Final accuracy: ", final_metrics["accuracy"])
    print("Final loss: ", final_metrics["loss"])
