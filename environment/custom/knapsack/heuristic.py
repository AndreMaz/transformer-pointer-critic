import sys
from typing import List, Tuple

sys.path.append('.')

from environment.custom.knapsack.backpack import Backpack
from environment.custom.knapsack.item import Item
from operator import itemgetter, attrgetter

import numpy as np

def solver(problem, num_backpacks: int):
    backpacks, items = parse_input(problem, num_backpacks)
    
    backpacks: List[Backpack] = sorted(backpacks, key=attrgetter("capacity"), reverse=False)
    items: List[Item] = sorted(items, key=lambda item: item.ratio, reverse=True)

    total_value = 0
    for backpack in backpacks:
        for item in items:
            if not item.is_taken():
                if item.weight + backpack.current_load <= backpack.capacity:
                    item.take() # Mark as taken

                    # Add to backpack. For stats
                    backpack.add_item(item.id, item.weight, item.value)
                    
                    # Add to the total value
                    total_value += item.value

    # print_backpack_stats(backpacks)

    return total_value

def parse_input(problem, num_backpacks) -> Tuple[List[Backpack], List[Item]]:
    backpacks = []
    for index, bps in enumerate(problem[:num_backpacks]):
        backpacks.append(
            Backpack(index, bps[0])
        )

    items = []
    for index, itm in enumerate(problem[num_backpacks:]):
        items.append(
            Item(
                index,
                weight = itm[0],
                value = itm[1]
            )
        )
    
    return backpacks, items

def print_backpack_stats(backpacks: List[Backpack]):
    for backpack in backpacks: backpack.print()

def validate_solution(backpacks: List[Backpack]):
    for backpack in backpacks:
        if backpack.is_valid() == False:
            return False

    return True

if __name__ == "__main__":

    problem = [[ 0.,  0.],
       [35.,  0.],
       [30.,  0.],
       [32.,  0.],
       [26.,  0.],
       [37.,  0.],
       [ 8., 62.],
       [ 5., 40.],
       [ 7., 61.],
       [ 5.,  7.],
       [10., 99.],
       [ 5., 53.],
       [10., 15.],
       [ 7., 85.],
       [ 5., 23.],
       [10., 82.],
       [ 8.,  4.],
       [ 8., 63.],
       [ 9., 74.],
       [ 7., 83.],
       [ 8., 73.],
       [ 5., 69.],
       [ 5., 18.],
       [ 5., 87.],
       [ 5., 80.],
       [10.,  8.]]
    num_backpacks = 6

    solver(problem, num_backpacks)