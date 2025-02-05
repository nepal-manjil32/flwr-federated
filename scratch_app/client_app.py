"""scratch-app: A Flower / PyTorch app."""
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import torch
from random import random
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from scratch_app.task import Net, get_weights, load_data, set_weights, test, train
import json

##-- Define Flower Client and client_fn --##
class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader, local_epochs):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

    ##-- Receives the parameters of the global model from the strategy (from sever app) --##
    def fit(self, parameters, config):
        set_weights(self.net, parameters) # applies those parameters to local model(self.net) of the client app
        #print(config)
        
        ##-- Training function to train the data locally -##
        train_loss = train(
            net = self.net,
            trainloader = self.trainloader,
            epochs = self.local_epochs,
            lr = config["lr"], # using the lr received from the strategy through config
            device = self.device,
        )

        complex_metric = {"abc": 123, "b": random(), "mylist": [1,2,3,4]}
        #!{"train_loss": train_loss, "random_num": complex_metric} this cannot be communicated as it violates metrics data structure
        complex_metric_str = json.dumps(complex_metric) # o/p we get is a string which is supported in metrics


        return (
            get_weights(self.net), # locally updated parameters
            len(self.trainloader.dataset), # size of dataset of the client
            {"train_loss": train_loss, "my_metric": complex_metric_str}, # metrics: train_loss data structure (dictionary) sent to the server
        )

    ##-- Receives the parameters of the global model and evaluates on the local data --##
    def evaluate(self, parameters, config):
        set_weights(self.net, parameters) # applies those parameters to local model(self.net) of the client app
        loss, accuracy = test(self.net, self.valloader, self.device)

        return loss, len(self.valloader.dataset), {"accuracy": accuracy}


def client_fn(context: Context):
    ##-- Load model and data --##
    net = Net()
    partition_id = context.node_config["partition-id"] # client id of the client app to be constructed (will be assigned at runtime any no between 0-9)
    num_partitions = context.node_config["num-partitions"] # how many clients are there in total in simulation (equal to num-supernodes)
    trainloader, valloader = load_data(partition_id, num_partitions)
    local_epochs = context.run_config["local-epochs"] # defined in pyproject.toml

    ##-- Return Client instance --##
    return FlowerClient(net, trainloader, valloader, local_epochs).to_client()


##-- Flower ClientApp --##
app = ClientApp(
    client_fn,
)
