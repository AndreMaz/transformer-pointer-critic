from environment.gym.env import GymEnvironment
from environment.custom.knapsack.env import Knapsack
from environment.custom.knapsack.env_v2 import KnapsackV2

custom_envs = {
    "Knapsack": Knapsack,
    "KnapsackV2": KnapsackV2,
}

def env_factory(type, name, opts):
    if (type == 'gym'): return GymEnvironment(name)

    #try:
    Environment = custom_envs[f"{name}"]
    return Environment(name, opts)
    # except KeyError:
    #    raise NameError(f'Unknown Environment Name! Select one of {list(custom_envs.keys())}')