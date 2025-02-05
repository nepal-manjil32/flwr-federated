##-- FedAvg is the basis for all the other starategies --##

from flwr.server.strategy import FedAvg
from flwr.server.client_proxy import ClientProxy
from flwr.common import FitRes, Parameters

class CustomFedAvg(FedAvg):

    # Constructor
    def __init__(self, *args, **kawrgs):
        super().__init__(*args, **kawrgs)
    
    def aggregate_fit(self, 
                      server_round: int, 
                      results: list[tuple[ClientProxy, FitRes]], 
                      failures: list[tuple[ClientProxy, FitRes] | BaseException]
                      ) -> tuple[Parameters | None , dict[str, bool | bytes | float | int | str]]:
        return super().aggregate_fit(server_round, results, failures)
    
