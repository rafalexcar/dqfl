"""fltabular: Flower Example on Adult Census Income Tabular Dataset."""

import torch
import numpy as np
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

from fltabular.task import (
    IncomeClassifier,
    evaluate,
    get_weights,
    load_data,
    set_weights,
    train,
)

def calculate_quality_metrics(loader, calculate_gini : bool=True, calculate_num_unique_classes : bool=True, calculate_avg_mi : bool=False, calculate_avg_mi_f : bool=False, calculate_f_accuracy : bool=False, calculate_label_counts : bool=True, calculate_label_counts_dic : bool=True):
    # Define required accumulators
    total_samples = 0  # For label distribution and feature accuracy
    running_feature_accuracy = 0.0  # Sum of feature accuracy (to calculate average later)
    mutual_info_sums = 0.0  # Accumulator for mutual information
    ff_info_sum = 0.0 # Accumulator for feature-feature information
    unique_label_set = set()  # Unique label tracker for num_unique_classes and gini
    label_counts_acc = {}  # Accumulator for label counts

    # Iterate through batches in the DataLoader
    for batch in loader:
        features = batch[0]  # Dataset-specific name for features
        labels = batch[1]

        # Track unique labels and counts
        if calculate_label_counts or calculate_gini or calculate_num_unique_classes:
            unique_labels, counts = torch.unique(labels, return_counts=True)

            # Update label counts (merge batch counts into global counts)
            for label, count in zip(unique_labels.tolist(), counts.tolist()):
                label_counts_acc[label] = label_counts_acc.get(label, 0) + count

            # Update unique label set
            unique_label_set.update(unique_labels.tolist())

    # Calculate Gini Index (if requested)
    if calculate_gini:
        total_label_count = sum(label_counts_acc.values())
        proportions = torch.tensor(list(label_counts_acc.values()), dtype=torch.float32) / total_label_count
        gini = 1 - torch.sum(proportions ** 2).item()
    else:
        gini = 0.0

    # Return the number of unique classes
    if calculate_num_unique_classes:
        num_unique_classes = len(unique_label_set)
    else:
        num_unique_classes = 0

    return {
        "gini": float(gini),
        "num_unique_classes": num_unique_classes
    }

class FlowerClient(NumPyClient):
    def __init__(self, partition_id, net, trainloader, testloader):
        self.net = net
        self.partition_id = partition_id
        self.trainloader = trainloader
        self.testloader = testloader

    def fit(self, parameters, config):
        set_weights(self.net, parameters)
        train_loss = train(self.net, self.trainloader)
        quality_metrics = calculate_quality_metrics(self.trainloader, calculate_avg_mi=False, calculate_gini=True, calculate_f_accuracy=False, calculate_avg_mi_f=False)
        class_balance = quality_metrics["gini"]
        if class_balance is None:
            raise ValueError("class balance is None! Check your fit function.")
        number_of_examples = len(self.trainloader.dataset)
        if number_of_examples is None:
            raise ValueError("number of examples is None! Check your fit function.")
        number_of_classes = quality_metrics["num_unique_classes"]
        if number_of_classes is None:
            raise ValueError("number of classes is None! Check your fit function.")


        return (
            get_weights(self.net),
            int(len(self.trainloader.dataset)), # Number of examples
            {"train_loss": train_loss,
             "class_balance": class_balance,
             "number_of_examples": number_of_examples,
             "number_of_classes": number_of_classes,
             "client_id" : self.partition_id},
        )


    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        loss, accuracy = evaluate(self.net, self.testloader)
        if loss is None:
            raise ValueError("loss is None! Check your evaluate function.")
        if accuracy is None:
            raise ValueError("accuracy is None! Check your evaluate function.")
        return loss, len(self.testloader), {"accuracy": accuracy}


def client_fn(context: Context):
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    local_epochs = context.run_config["local-epochs"]

    train_loader, test_loader = load_data(
        partition_id=partition_id, num_partitions=num_partitions, context=context
    )
    net = IncomeClassifier()
    return FlowerClient(partition_id=partition_id, net=net, trainloader=train_loader, testloader=test_loader).to_client()


app = ClientApp(client_fn=client_fn)
