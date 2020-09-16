import sys
from typing import List, Tuple

sys.path.append('.')

from environment.custom.knapsack.backpack import Backpack
from environment.custom.knapsack.item import Item
from operator import itemgetter, attrgetter

# More info about the heuristic here: https://github.com/jmyrberg/mknapsack
from mknapsack.algorithms import mtm

import numpy as np

def solver(problem, num_backpacks: int):
    backpacks, item_weights, item_values = parse_input(problem, num_backpacks)

    total_value, x, bt, glopt = mtm(item_values, item_weights, backpacks)
    
    return total_value

def parse_input(problem, num_backpacks) -> Tuple[List[Backpack], List[Item]]:
    backpacks = []
    for index, bps in enumerate(problem[:num_backpacks]):
        backpacks.append(int(bps[0]))

    item_weights = []
    item_values = []
    for index, itm in enumerate(problem[num_backpacks:]):
        item_weights.append(int(itm[0]))
        item_values.append(int(itm[1]))
    
    return backpacks, item_weights, item_values