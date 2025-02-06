"""scratch-app: A Flower / PyTorch app."""
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from flwr.common import Context, ndarrays_to_parameters, Metrics
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from scratch_app.task import Net, get_weights , set_weights, test, get_transforms
from typing import List, Tuple
from datasets import load_dataset
from torch.utils.data import DataLoader
from random import random
import json
from scratch_app.my_strategy import CustomFedAvg

##-- Callback(function) to perform centralized evaluation of our model --##
 # called each time our global model is updated
 # after all the updates from each client is aggregated
def get_evaluate_fn(testloader, device):
    """ Return a callback that evaluates the global model"""
    def evaluate(server_round, 
                 parameters_ndarray, # parameters of global model
                 config):
        """ Evaluate global model using provided centralized dataset """
        net = Net() # instantiate the pytroch model you want to evalute
        set_weights(net, parameters_ndarray)
        net.to(device)
        loss, accuracy = test(net, testloader, device)

        return loss, {"centralized_accuracy": accuracy}
    
    return evaluate



##-- Callback(function) to aggregate all the accuracy received from the clients --##
 # not a centralized accuracy
 # got by running the global model in a distributed fashion in every client app
 # every client receive the same global model and evaluate it one their repsective data and report the accuracies after each round
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """ A function that aggregates metrics """
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics] # weighted_accuracy
    total_examples = sum(num_examples for num_examples, _ in metrics)

    """
    Example metrics: 
        #! List of tuples
        metrics = [
            (10, {"accuracy": 0.8}),  # 10 examples with 80% accuracy
            (15, {"accuracy": 0.9}),  # 15 examples with 90% accuracy
            (20, {"accuracy": 0.75})  # 20 examples with 75% accuracy
        ]

        #! Calculate num_examples * m["accuracy"] for each tuple
        result = [num_examples * m["accuracy"] for num_examples, m in metrics]

        print(result)  # Output: [8.0, 13.5, 15.0]
    """

    """
    Example: 
        #! List of tuples
        metrics = [(5, 0.8), (10, 0.9), (15, 0.7)]

        #! Use `_` to ignore the second value in each tuple
        total = sum(num_examples for num_examples, _ in metrics)

        print(total)  # Output: 30 (5 + 10 + 15)
    """

    return {"weighted_accuracy": sum(accuracies) / total_examples}




##-- Callback(function) for fit method of client --##
def handle_fit_metrics(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """ Handle metrics (received after each round) from fit method in clients """
    b_values = []
    for _, m in metrics:
        my_metric_str = m['my_metric']
        print(my_metric_str)
        my_metric = json.loads(my_metric_str) # convert back to dictionary
        b_values.append(my_metric['b'])

    return {"max_b": max(b_values)}

##-- Callback(function) that will be called internally by the strategy when configuring a round of fit --##
    # configure_fit: a set of clients will be sampled and some instructions are prepared for each client which includes
        # 1. the model
        # 2. a config that can sent optionally to the client
def on_fit_config(server_round: int) -> Metrics:
    """ Adjust learning rate based on the current round """
    # New hyperparameters not defined previously
    lr = 0.01
    if server_round > 2:
        lr = 0.05

    return {"lr": lr}


def server_fn(context: Context):
    ##-- Read from config --##
    num_rounds = context.run_config["num-server-rounds"] # defined in pyproject.toml
    fraction_fit = context.run_config["fraction-fit"] # defined in pyproject.toml

    ##-- Initialize model parameters --##
    ndarrays = get_weights(Net())
    parameters = ndarrays_to_parameters(ndarrays) # convert NumPy ndarrays to parameters object.

    ## -- Load a global test set (Update this part of the code if you want to load your custom dataset made using torchvision) --##
    testset = load_dataset("zalando-datasets/fashion_mnist")["test"]
    testloader = DataLoader(testset.with_transform(get_transforms()), batch_size=32)

    ##-- Define strategy (brain of the server app) --##
    strategy = FedAvg(
        fraction_fit=fraction_fit, # what's the percentage of client we are gonna sample each round (refer above 'Read from config')
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,

        ##-- Making use of callbacks --##
        evaluate_metrics_aggregation_fn=weighted_average, # callback to aggregate the metrics returned by the evaluate method of client
        fit_metrics_aggregation_fn=handle_fit_metrics,   # callback to aggregate the metrics returned by the fit method of client
        on_fit_config_fn=on_fit_config, # passing to the fit function
        evaluate_fn=get_evaluate_fn(testloader, device="cpu") # perform centralized evaluation of our global model (called each time our global model is updated)
    )
    config = ServerConfig(num_rounds=num_rounds) # primarily used to specify the number of rounds
    return ServerAppComponents(strategy=strategy, config=config)


##-- Create ServerApp --##
app = ServerApp(server_fn=server_fn)
