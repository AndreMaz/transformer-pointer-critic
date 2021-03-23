import sys

import numpy as np
sys.path.append('.')
import json
import random

from typing import List, Tuple

from environment.custom.knapsack_v2.heuristic.base_heuristic import BaseHeuristic
from environment.custom.knapsack_v2.bin import Bin
from environment.custom.knapsack_v2.item import Item
from operator import itemgetter, attrgetter

class RandomHeuristic(BaseHeuristic):
    def __init__(self,
                num_nodes: int,
                opts: dict
                ):
        super(RandomHeuristic, self).__init__(num_nodes)

        self.generate_name()
    
    def generate_name(self):
        self.name = 'random'

    def solve(self, state):
        
        bin_list = self.parse_bins(state)
        EOS_NODE = bin_list.pop(0)

        item_list = self.parse_items(state)
        
        while len(item_list) > 0:
            # Randomly pick an item
            item_index = random.randrange(len(item_list))
            item = item_list.pop(item_index)

            copy_list = [ ] + bin_list

            self.place_single_item(item, copy_list, EOS_NODE)
            
        # Store a reference with the solution
        self.solution = [EOS_NODE] + bin_list
    
    def place_single_item(self, item, bin_list: List[Bin], EOS_NODE: Bin):
        
        # Now do the fit first
        allocated = False
        while len(bin_list) > 0:
            # Randomly pick a node
            node_index = random.randrange(len(bin_list))
            node: Bin = bin_list.pop(node_index)

            if node.can_fit_item(item):
                node.insert_item(item)
                allocated = True
                break

        if not allocated:
            # Place at EOS node
            EOS_NODE.insert_item(item)

if __name__ == "__main__":
    with open(f"configs/KnapsackV2.json") as json_file:
        params = json.load(json_file)

    heuristic_opts = params['tester_config']['heuristic']['random']

    dummy_state = np.array([
        [
            [-2, -2],
            [ 0.1,  0.0],
            [ 0.5,  0.0],
            [ 0.2,  0.1],
            [ 0.3,  0.5],
        ]
    ], dtype='float32')
    
    node_sample_size = 3
    
    solver = RandomHeuristic(node_sample_size, heuristic_opts)

    solver.solve(dummy_state)

    solver.solution