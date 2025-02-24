"""dqfl: A Flower / PyTorch app."""
from typing import List, Tuple, Dict
from flwr.common import Context, ndarrays_to_parameters, Metrics
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
import pandas as pd
from dqfl.task import Net, get_weights
from dqfl.my_strategy import CustomFedAvg
import math
import json
import statistics
import logging


# Initialize label dictionary on the server
label_occurrences = {}  # {label: total_count}
# Define a log file for result metrics
log_file = "experiment_log.json"

# Set up logging to the file
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,  # Log level can be adjusted
    format="%(message)s",
)

def euclidean_distance(client_labels: dict, avg_occurrences: dict) -> float:
    """Compute Euclidean distance between client labels and average occurrences,
    considering missing labels as 0."""
    all_labels = set(client_labels.keys()).union(set(avg_occurrences.keys()))  # Combine all unique labels
    distance = 0.0
    for label in all_labels:
        client_count = client_labels.get(label, 0)  # Default to 0 if label not present
        avg_count = avg_occurrences.get(label, 0)
        distance += (client_count - avg_count) ** 2
    return math.sqrt(distance)


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """A function that aggregates the accuracies of each local client (evaluate metrics aggregation function)."""
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    total_examples = sum(num_examples for num_examples, _ in metrics)
    avg_accuracy = sum(accuracies) / total_examples

    result = {
        "avr_accuracy": avg_accuracy, #for image classification, accuracy makes sense
    }

    # Log the accuracy values
    logging.info(json.dumps(result))  # Log as JSON

    return result

def handle_fit_metrics(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Handle metrics from fit method in clients."""
    global label_occurrences # Check the best way to do that

    class_balances = [] # Initiate an empty list to hold class balances over rounds
    num_examples = 0.0
    num_classes = 0.0
    mutual_info = 0.0
    feature_accuracy = 0.0
    label_distribution_distance = 0.0
    ff_info = 0.0

    for _, m in metrics:
        client_labels_str = m["label_counts"] # To get the dict {label: count} as a string

        # revert client_labels from json to dictionary
        client_labels = json.loads(client_labels_str)

        #update global label counts
        for label, count in client_labels.items():
            if label in label_occurrences:
                label_occurrences[label] += count
            else:
                label_occurrences[label] = count
        # Compute average occurrences per label
        #average_label_occurrences = {label: count / len(metrics) for label, count in label_occurrences.items()}

        # Calculate the euclidean distance between average_label_occurrences and client_labels
        #labels_distance = euclidean_distance(client_labels, average_label_occurrences)
        #print(f"labels_distance: {labels_distance}")

        # Define label distribution balance as being higher, when labels_distance is smaller
        # Summarize for each round, over all clients
        #normalization_factor = sum(average_label_occurrences.values())
        #label_distribution_balance = max(0, 1 - labels_distance / normalization_factor)
        #print(f"normalization_factor: {normalization_factor}")

        #print(f"label_distribution_balance: {label_distribution_balance}")


        # handle other metrics across the rounds
        class_balances = [(m["client_id"], m["class_balance"]) for _, m in metrics]
        num_examples = [(m["client_id"], m["number_of_examples"]) for _, m in metrics]
        num_classes = [(m["client_id"],m["number_of_classes"]) for _, m in metrics]
        mutual_info = [(m["client_id"],m["avg_mi"]) for _, m in metrics]
        feature_accuracy = [(m["client_id"],m["feature_accuracy"]) for _, m in metrics]
        label_distribution_distance = [(m["client_id"], m["label_distribution_distance"]) for _, m in metrics]
        ff_info = [(m["client_id"],m["avg_ffi"]) for _, m in metrics]


    return {"class_balance": class_balances,
            "num_examples": num_examples,
            "num_classes": num_classes,
            "mutual_infor": mutual_info,
            "feature_accuracy": feature_accuracy,
            "label_distribution_distance": label_distribution_distance,
            "ff_info": ff_info}

def on_fit_config(server_round: int) -> Metrics:
    """Generate config for each round."""
    global label_occurrences  # Assume this is maintained globally on the server

    # Compute average label occurrences
    total_clients = max(5 * server_round, 1)  # Avoid division by zero
    average_label_occurrences = {label: count / total_clients for label, count in label_occurrences.items()}

    """Adjusts learning rate."""
    lr = 0.01

    return {"lr": lr,
            "average_label_occurrences": json.dumps(average_label_occurrences)
            }



def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"] #percentage of clients for each round

    # Initialize model parameters
    ndarrays = get_weights(Net())
    parameters = ndarrays_to_parameters(ndarrays)

    # Define strategy
    strategy = CustomFedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
        evaluate_metrics_aggregation_fn=weighted_average,
        fit_metrics_aggregation_fn=handle_fit_metrics,
        on_fit_config_fn=on_fit_config,
        context=context,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
