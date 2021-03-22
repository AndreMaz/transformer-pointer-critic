from typing import Callable, List, Tuple
from environment.gym.env import GymEnvironment

from environment.custom.knapsack_v2.env import KnapsackEnvironmentV2
from environment.custom.knapsack_v2.tester import test as KnapsackV2Tester


from environment.custom.vrp.env import CVRP
# from environment.custom.vrp.optimum_solver import solver as CVRPSolver
# from environment.custom.vrp.heuristic import solver as CVRPHeuristic

from environment.custom.resource.env import ResourceEnvironment
from environment.custom.resource.heuristic import GreedyHeuristic as GreedyHeuristicV1
from environment.custom.resource.tester import test as ResourceTester

from environment.custom.resource_v3.env import ResourceEnvironmentV3
from environment.custom.resource_v3.heuristic.factory import heuristic_factory
from environment.custom.resource_v3.tester import test as ResourceV3Tester

custom_envs = {
    "KnapsackV2": (KnapsackEnvironmentV2, KnapsackV2Tester),
    "CVRP": (CVRP, None),
    "Resource": (ResourceEnvironment, ResourceTester),
    "ResourceV3": (ResourceEnvironmentV3, ResourceV3Tester),
}

def env_factory(type, name, opts) -> Tuple[ResourceEnvironmentV3, Callable, Callable]:
    if (type == 'gym'): return GymEnvironment(name)

    #try:
    Environment, tester = custom_envs[f"{name}"]
    # print(f'"{name.upper()}" environment selected.')
    return Environment(name, opts), tester
    # except KeyError:
    #    raise NameError(f'Unknown Environment Name! Select one of {list(custom_envs.keys())}')