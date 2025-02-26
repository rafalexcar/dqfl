from symbol import parameters
from typing import Union, Optional
from flwr.server import ClientManager
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from flwr.server.strategy.aggregate import aggregate
from torch.cuda import seed_all
from .task import IncomeClassifier, set_weights
from functools import partial, reduce
from flwr.common import Context
import torch
import sys
import json
import numpy as np
import random



# Define dq weights equally
def define_weights_equally(metric_names: list[str]):
    dq_weights = {}
    for metric in metric_names:
        dq_weights[metric] = 1.0 / len(metric_names)
    return dq_weights

def aggregate_normalized(results: list[tuple[NDArrays, float]]) -> NDArrays:
    """Compute weighted average with normalized weights."""
    # Compute the total sum of weights
    total_weight = sum(weight for _, weight in results)

    # Normalize the weights so they sum to 1
    normalized_weights = [(weights, weight / total_weight) for weights, weight in results]

    # Compute the weighted average using the normalized weights
    aggregated_ndarrays: NDArrays = [
        reduce(np.add, [layer * weight for layer, (_, weight) in zip(weights, normalized_weights)])
        for weights in zip(*[r[0] for r in normalized_weights])
    ]

    return aggregated_ndarrays


class CustomFedAvg(FedAvg):

    def __init__(self, context : Context, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Default weights if none provided
        self.weights = {"class_balance": context.run_config['class_balance'],
                        "num_examples": context.run_config['num_examples'],
                        "num_classes": context.run_config['num_classes']}

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        """Aggregate fit results with custom weighting."""
        if not results:
            return None, {}

        # Define a function to compute the combined score for each client
        def compute_score(
                fit_res: FitRes,
                max_num_examples: int,
                max_num_classes: int,
                weights: dict[str, float],
        ) -> float:
            """Compute the quality score for a client."""

            # Extract metrics
            class_balance = fit_res.metrics["class_balance"]
            num_examples = fit_res.metrics["number_of_examples"]
            num_classes = fit_res.metrics["number_of_classes"]


            # Normalize metrics
            norm_class_balance = class_balance  # Already between 0 and 1
            norm_num_examples = num_examples / max_num_examples if max_num_examples > 0 else 0
            norm_num_classes = num_classes / max_num_classes if max_num_classes > 0 else 0


            # Compute weighted score
            score = (
                    weights["class_balance"] * norm_class_balance +
                    weights["num_examples"] * norm_num_examples +
                    weights["num_classes"] * norm_num_classes
            )

            return score

        # Precompute max and min values
        max_num_examples = max(r[1].metrics["number_of_examples"] for r in results)
        max_num_classes = max(r[1].metrics["number_of_classes"] for r in results)


        # This was a try to construct a function to automatically equally attribute the weights
        #metrics = ["class_balance", "num_examples", "num_classes", "avg_mi", "feature_accuracy", "label_distribution_balance"]
        #weights = define_weights_equally(metrics)
        weights = self.weights

        # Compute adjusted weights using quality score and number of examples
        weights_results = []
        for client_proxy, fit_res in results:
            num_examples = fit_res.num_examples
            quality_score = compute_score(fit_res, max_num_examples, max_num_classes, weights)
            alpha = 0.0 # Parameter to control the contribution of new metrics : alpha = 1.0 => no contribution; alpha = 0.0 => max contribution
            adjusted_weight = num_examples * (alpha + (1 - alpha) * quality_score)
            weights_results.append((parameters_to_ndarrays(fit_res.parameters), adjusted_weight))

        # Aggregate using the adjusted weights
        aggregated_ndarrays = aggregate_normalized(weights_results)
        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        # Aggregate custom metrics if a function is provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)

        # Save the global model after aggregation
        model = IncomeClassifier()
        set_weights(model, aggregated_ndarrays)
        torch.save(model.state_dict(), f"global_model_{server_round}")

        return parameters_aggregated, metrics_aggregated

    # def configure_fit(
    #         self, server_round: int, parameters: Parameters, client_manager: ClientManager
    # ) -> list[tuple[ClientProxy, FitIns]]:
    #     """Configure the next round of training with reproducible client selection."""
    #
    #     # Create fit configuration
    #     config = {}
    #     if self.on_fit_config_fn is not None:
    #         config = self.on_fit_config_fn(server_round)
    #     fit_ins = FitIns(parameters, config)
    #
    #     # Sample clients reproducibly using a fixed seed
    #     sample_size, min_num_clients = self.num_fit_clients(client_manager.num_available())
    #
    #     # Set a fixed seed for reproducibility
    #     random.seed(42 + server_round)  # Adding server_round ensures different clients each round if needed
    #
    #     # Get all available clients and sort by client ID
    #     all_clients = list(client_manager.all().values())
    #     sorted_clients = sorted(all_clients, key=lambda client: client.cid)
    #
    #     # Sample reproducibly from sorted clients
    #     clients = random.sample(sorted_clients, sample_size)
    #
    #     return [(client, fit_ins) for client in clients]