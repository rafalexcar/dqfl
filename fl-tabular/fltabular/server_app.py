"""fltabular: Flower Example on Adult Census Income Tabular Dataset."""

from flwr.common import ndarrays_to_parameters
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
from flwr.common import Context
from typing import List, Tuple, Dict, Union
from flwr.common import Context, ndarrays_to_parameters, Metrics
from fltabular.task import IncomeClassifier, get_weights
from .my_strategy import CustomFedAvg


def weighted_average(metrics):
    """A function that aggregates the accuracies of each local client (evaluate metrics aggregation function)."""
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    total_examples = sum(num_examples for num_examples, _ in metrics)
    avg_accuracy = sum(accuracies) / total_examples

    result = {
        "avr_accuracy": avg_accuracy, #for image classification, accuracy makes sense
    }

    return result

def handle_fit_metrics(metrics: List[Tuple[int, Metrics]]) -> Metrics:
        num_examples = 0.0
        num_classes = 0.0
        class_balance = 0.0

        for _, m in metrics:
            # handle other metrics across the rounds
            num_examples = [(m["client_id"], m["number_of_examples"]) for _, m in metrics]
            num_classes = [(m["client_id"],m["number_of_classes"]) for _, m in metrics]
            class_balance = [(m["client_id"],m["class_balance"]) for _, m in metrics]


        return {"num_examples": num_examples,
                "num_classes": num_classes,
                "class_balance": class_balance
                }





def server_fn(context: Context) -> ServerAppComponents:
    net = IncomeClassifier()
    params = ndarrays_to_parameters(get_weights(net))
    fraction_fit = context.run_config["fraction-fit"] #percentage of clients for each round


    strategy = CustomFedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=params,
        evaluate_metrics_aggregation_fn=weighted_average,
        fit_metrics_aggregation_fn=handle_fit_metrics,
        context=context,
    )
    num_rounds = context.run_config["num-server-rounds"]
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(config=config, strategy=strategy)


app = ServerApp(server_fn=server_fn)
