"""dqfl: A Flower / PyTorch app."""

import torch
import numpy as np
import json
from random import random
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from numpy import ndarray
from dqfl.server_app import euclidean_distance
from dqfl.task import Net, get_weights, load_data, set_weights, test, train
from collections import Counter
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import normalized_mutual_info_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd


def ff_information(X):
    num_bins = int(np.sqrt(len(X)))
    features = X.columns
    num_features = len(features)

    # Initialize an empty DataFrame to store mutual information values
    mi_matrix = pd.DataFrame(np.zeros((num_features, num_features)), columns=features, index=features)

    # Preprocess each feature to ensure that all features are discrete (integers)
    processed_X = X.copy()
    encoders = {}

    for feature in features:
        if X[feature].dtype == 'object' or X[feature].dtype.name == 'category':
            # Encode categorical features using LabelEncoder
            le = LabelEncoder()
            processed_X[feature] = le.fit_transform(X[feature].astype(str))  # Encode as integers
            encoders[feature] = le
        elif np.issubdtype(X[feature].dtype, np.number):  # Check for numerical features
            # Discretize numerical features using binning (with dynamic bin count, e.g., Sturges' rule)
            num_bins = int(np.ceil(np.log2(len(X[feature])) + 1))  # Sturges' rule for bin count
            processed_X[feature] = pd.cut(X[feature], bins=num_bins, labels=False).astype(int)

    # Calculate mutual information for each feature pair
    for i in range(num_features):
        for j in range(i + 1, num_features):  # Avoid redundant calculations (symmetry)
            mi_value = normalized_mutual_info_score(processed_X[features[i]], processed_X[features[j]])
            mi_matrix.iloc[i, j] = mi_value
            mi_matrix.iloc[j, i] = mi_value  # Symmetric assignment

    # Compute the Average Information Score
    num_comparisons = (num_features * (num_features - 1)) / 2  # Upper triangle count
    average_info_score = mi_matrix.sum().sum() / (2 * num_comparisons)  # Sum divided by num comparisons
    return average_info_score


def entropy(labels):
    _, counts = torch.unique(labels, return_counts=True)
    probabilities = counts.float() / labels.size(0)
    return -torch.sum(probabilities * torch.log2(probabilities))


def mutual_information(features, labels):
    H_Y = entropy(labels)
    _, H_XY = torch.unique(features, return_counts=True)
    return H_Y - H_XY

def calculate_feature_accuracy(all_features, num_samples, lower_quantile=0.05, upper_quantile=0.95):
    """
    Calculate feature accuracy as the degree of outliers' presence feature-wise using quantiles.

    Args:
        all_features (torch.Tensor): Tensor of shape [num_samples, num_features].
        lower_quantile (float): Lower quantile threshold (default: 0.05).
        upper_quantile (float): Upper quantile threshold (default: 0.95).

    Returns:
        float: Average feature accuracy across all features.
    """

    # Calculate lower and upper quantiles for each feature
    lower_bounds = torch.quantile(all_features, lower_quantile, dim=0)
    upper_bounds = torch.quantile(all_features, upper_quantile, dim=0)

    # Count outliers for each feature (values outside the quantile range)
    outliers = (all_features < lower_bounds) | (all_features > upper_bounds)
    num_outliers_per_feature = outliers.sum(dim=0)

    # Calculate feature accuracy for each feature
    feature_accuracies = 1 - (num_outliers_per_feature / num_samples)

    # Compute the average feature accuracy over all features
    avg_feature_accuracy = torch.mean(feature_accuracies)

    return avg_feature_accuracy

def calculate_quality_metrics(loader, calculate_gini : bool=False, calculate_num_unique_classes : bool=True, calculate_avg_mi : bool=False, calculate_avg_mi_f : bool=False, calculate_f_accuracy : bool=False, calculate_label_counts : bool=True, calculate_label_counts_dic : bool=True):

    # Define required accumulators
    total_samples = 0  # For label distribution and feature accuracy
    running_feature_accuracy = 0.0  # Sum of feature accuracy (to calculate average later)
    mutual_info_sums = 0.0  # Accumulator for mutual information
    ff_info_sum = 0.0 # Accumulator for feature-feature information
    unique_label_set = set()  # Unique label tracker for num_unique_classes and gini
    label_counts_acc = {}  # Accumulator for label counts

    # Iterate through batches in the DataLoader
    for batch in loader:
        features = batch['image']  # Dataset-specific name for features
        labels = batch["label"]

        batch_size = labels.size(0)
        total_samples += batch_size

        # Process Feature Accuracy if requested
        if calculate_f_accuracy:
            avg_feature_accuracy = calculate_feature_accuracy(
                features.view(features.size(0), -1),  # Flatten features to 2D
                batch_size,
                lower_quantile=0.05,
                upper_quantile=0.95
            )
            running_feature_accuracy += avg_feature_accuracy * batch_size  # Weighted sum

        # Process Feature-Target Information if requested
        if calculate_avg_mi:
            mutual_info = mutual_information(
                features.view(features.size(0), -1),  # Flatten features
                labels
            )
            mutual_info_sums += mutual_info.sum().item()  # Sum of batch mutual info

        # Process Feature-Feature Information if requested
        if calculate_avg_mi_f:
            features_df = pd.DataFrame(features.view(features.size(0), -1).numpy())
            mutual_info_f = ff_information(features_df) #Requires df
            ff_info_sum += mutual_info_f

        # Track unique labels and counts
        if calculate_label_counts or calculate_gini or calculate_num_unique_classes:
            unique_labels, counts = torch.unique(labels, return_counts=True)

            # Update label counts (merge batch counts into global counts)
            for label, count in zip(unique_labels.tolist(), counts.tolist()):
                label_counts_acc[label] = label_counts_acc.get(label, 0) + count

            # Update unique label set
            unique_label_set.update(unique_labels.tolist())

    # After looping through batches, process the accumulated metrics:

    # Calculate Feature Accuracy (weighted average over all batches)
    if calculate_f_accuracy:
        final_avg_feature_accuracy = running_feature_accuracy / total_samples
    else:
        final_avg_feature_accuracy = 0.0

    # Calculate Average Mutual Information
    if calculate_avg_mi:
        avg_mi = mutual_info_sums / total_samples  # Average MI
        max_mi = torch.log2(
            torch.tensor(len(unique_label_set), dtype=torch.float32))  # Maximum MI (based on unique labels)
        normalized_avg_mi = avg_mi / max_mi if max_mi > 0 else 0.0
    else:
        normalized_avg_mi = 0.0

    # Calculate Average Feature-Feature Information (already normalized)
    if calculate_avg_mi_f:
        avg_ff_info = ff_info_sum/ total_samples
    else:
        avg_ff_info = 0.0

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

    # Serialize label counts dictionary if required
    if calculate_label_counts:
        label_counts_str = json.dumps(label_counts_acc)  # Convert dict to JSON string
    else:
        label_counts_str = ""

    return {
        "gini": float(gini),
        "num_unique_classes": num_unique_classes,
        "avg_mi": float(normalized_avg_mi),
        "feature_accuracy": float(final_avg_feature_accuracy),
        "label_counts": label_counts_str,  # Pass the dict as a string
        "label_counts_dic": label_counts_acc,  # Pass as raw dictionary
        "avg_ffi": float(avg_ff_info),
    }


def calculate_distributed_quality_metrics(label_counts, average_label_occurrences, current_round):
    """Calculate the normalized label distribution balance metric."""
    if current_round == 1:
        return 1.0  # Perfect balance for the initial round

    # Compute the distance to the average label occurrences
    distance_to_average = euclidean_distance(label_counts, average_label_occurrences)

    return {"label_distribution_distance" : distance_to_average} # Don't forget to check normalization


# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    def __init__(self, partition_id, net, trainloader, valloader, local_epochs):
        self.partition_id = partition_id
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)



    def fit(self, parameters, config):
        set_weights(self.net, parameters)
        print(config)
        train_loss = train(
            self.net,
            self.trainloader,
            self.local_epochs,
            config["lr"],
            self.device,
        )

        quality_metrics = calculate_quality_metrics(self.trainloader, calculate_avg_mi=False, calculate_gini=False, calculate_f_accuracy=False, calculate_avg_mi_f=False)
        gini_index = quality_metrics["gini"]
        class_balance = gini_index
        number_of_examples = len(self.trainloader.dataset)
        number_of_classes = quality_metrics["num_unique_classes"]
        avg_mi = quality_metrics["avg_mi"]
        feature_accuracy = quality_metrics["feature_accuracy"]
        label_counts = quality_metrics["label_counts"]
        label_counts_dic = quality_metrics["label_counts_dic"]
        avg_ffi = quality_metrics["avg_ffi"]

        #  Check if it is possible to assess returned metrics from the server using the config
        current_round = config.get("lr", 1) # Get the current round to handle the round 0 for label distribution balance
        average_label_occurrences = json.loads(config.get("average_label_occurrences", "{}"))

        print(f"Received average label occurrences: {average_label_occurrences}")
        print(f"current label counts: {label_counts}")
        label_distribution_distance = calculate_distributed_quality_metrics(label_counts_dic, average_label_occurrences, current_round)["label_distribution_distance"]
        print(f"label distribution distance: {label_distribution_distance}")

        return (
            get_weights(self.net),
            int(len(self.trainloader.dataset)), # Number of examples
            {"train_loss": train_loss,
             "class_balance": class_balance,
             "number_of_examples": number_of_examples,
             "number_of_classes": number_of_classes,
             "avg_mi": avg_mi,
             "feature_accuracy": feature_accuracy,
             "label_counts": label_counts,
             "label_distribution_distance": label_distribution_distance,
             "avg_ffi": avg_ffi,
             "client_id" : self.partition_id},
        )

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader, self.device) # communicate evaluation metrics
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}


def client_fn(context: Context):
    # Load model and data
    net = Net()
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, valloader = load_data(partition_id, num_partitions, context) #num_partitions = clients in total
    local_epochs = context.run_config["local-epochs"]

    # Return Client instance
    return FlowerClient(partition_id=partition_id,net=net, trainloader=trainloader, valloader=valloader, local_epochs=local_epochs).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn,
)
