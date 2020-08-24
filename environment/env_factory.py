from environment.gym.env import GymEnvironment

from environment.custom.knapsack.env_v2 import KnapsackV2
from environment.custom.knapsack.optimum_solver import solver as KnapsackSolver
from environment.custom.knapsack.heuristic import solver as KnapsackHeuristic

custom_envs = {
    "KnapsackV2": (KnapsackV2, KnapsackSolver, KnapsackHeuristic),
}

def env_factory(type, name, opts):
    if (type == 'gym'): return GymEnvironment(name)

    #try:
    Environment, optimum_solver, heuristic = custom_envs[f"{name}"]
    return Environment(name, opts), optimum_solver, heuristic
    # except KeyError:
    #    raise NameError(f'Unknown Environment Name! Select one of {list(custom_envs.keys())}')