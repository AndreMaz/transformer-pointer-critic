from typing import List
from environment.gym.env import GymEnvironment

from environment.custom.knapsack.env_v2 import KnapsackV2
from environment.custom.knapsack.optimum_solver import solver as KnapsackSolver
from environment.custom.knapsack.heuristic import solver as KnapsackHeuristic
from environment.custom.knapsack.tester import test as KnapsackTester
from environment.custom.knapsack.plotter import plotter as KnapsackPlotter

from environment.custom.vrp.env import CVRP
# from environment.custom.vrp.optimum_solver import solver as CVRPSolver
# from environment.custom.vrp.heuristic import solver as CVRPHeuristic

from environment.custom.resource.env import ResourceEnvironment
from environment.custom.resource.heuristic import GreedyHeuristic as GreedyHeuristicV1
from environment.custom.resource.tester import test as ResourceTester
from environment.custom.resource.plotter import plotter as ResourcePlotter

from environment.custom.resource_v3.env import ResourceEnvironmentV3
from environment.custom.resource_v3.heuristic.factory import heuristic_factory
from environment.custom.resource_v3.tester import test as ResourceV3Tester
from environment.custom.resource_v3.plotter import plotter as ResourceV3Plotter


custom_envs = {
    "KnapsackV2": (KnapsackV2, KnapsackTester, KnapsackPlotter),
    "CVRP": (CVRP, None, None),
    "Resource": (ResourceEnvironment, ResourceTester, ResourcePlotter),
    "ResourceV3": (ResourceEnvironmentV3, ResourceV3Tester, ResourceV3Plotter),
}

def env_factory(type, name, opts):
    if (type == 'gym'): return GymEnvironment(name)

    #try:
    Environment, tester, plotter = custom_envs[f"{name}"]
    # print(f'"{name.upper()}" environment selected.')
    return Environment(name, opts), tester, plotter
    # except KeyError:
    #    raise NameError(f'Unknown Environment Name! Select one of {list(custom_envs.keys())}')