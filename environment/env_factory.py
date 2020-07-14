from environment.gym.env import GymEnvironment
from environment.custom.fill_the_cells.fill_the_cells import FillTheCells
from environment.custom.sorting_numbers.sorting import SortingEnvironment
from environment.custom.knapsack.env import Knapsack

custom_envs = {
    "Fill-The-Cells": FillTheCells,
    "Sorting-Numbers": SortingEnvironment,
    "Knapsack": Knapsack
}

def env_factory(type, name, opts):
    if (type == 'gym'): return GymEnvironment(name)

    #try:
    Environment = custom_envs[f"{name}"]
    return Environment(name, opts)
    # except KeyError:
    #    raise NameError(f'Unknown Environment Name! Select one of {list(custom_envs.keys())}')