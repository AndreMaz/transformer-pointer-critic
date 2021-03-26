import sys

import numpy as np
sys.path.append('.')
import json

from typing import List, Tuple

from environment.custom.knapsack_v2.heuristic.base_heuristic import BaseHeuristic
from environment.custom.knapsack_v2.bin import Bin
from environment.custom.knapsack_v2.item import Item
from operator import itemgetter, attrgetter

class WasteReductionHeuristic(BaseHeuristic):
    def __init__(self,
                num_nodes: int,
                opts: dict
                ):
        super(WasteReductionHeuristic, self).__init__(num_nodes)

        self.item_sort_descending = opts['item_sort_descending']
        self.bin_sort_descending = opts['bin_sort_descending']

        self.generate_name()
    
    def generate_name(self):
        self.name = f'item_ASC_{self.item_sort_descending}_bin_ASC_{self.bin_sort_descending}'

    def solve(self, state):
        
        bin_list = self.parse_bins(state)
        EOS_NODE = bin_list.pop(0)

        item_list = self.parse_items(state)
        
        # Sort the resources in a descending order
        item_list: List[Item] = sorted(
            item_list,
            key=resource_sorting_fn,
            reverse=self.item_sort_descending
        )
        
        for item in item_list:
            self.place_single_resource(item, bin_list, EOS_NODE)
            
        # Store a reference with the solution
        self.solution = [EOS_NODE] + bin_list
    
    def place_single_resource(self, item: Item, bin_list: List[Bin], EOS_NODE: Bin):
        diffs = compute_potential_placement_diffs(item, bin_list)

        # Sort the nodes by dominant resource
        sorted_bins: Tuple[float, Bin] = sorted(
            diffs,
            key=node_sorting_fn,
            reverse=self.bin_sort_descending
        )

        # Now do the fit first
        allocated = False
        bin: Bin
        for diff, bin in sorted_bins:
            if (diff >= 0):
                bin.insert_item(item)
                allocated = True
                break

        if not allocated:
            # Place at EOS node
            EOS_NODE.insert_item(item)

def compute_potential_placement_diffs(item, bin_list) -> Tuple[float, Bin]:
        # Compute dominant resource of each node and current request
        diffs = []
        bin: Bin
        for bin in bin_list:
            diffs.append( 
                (bin.compute_remaining_capacity(item), bin)
            )

        return diffs

def node_sorting_fn(e: Tuple[float, Bin]):
    return e[0]
    # return (node.remaining_CPU, node.remaining_RAM, node.remaining_MEM)

def resource_sorting_fn(elem: Item):
    return elem.ratio
    
if  __name__ == "__main__": # pragma: no cover
    with open(f"configs/KnapsackV2.json") as json_file:
        params = json.load(json_file)

    heuristic_opts = params['tester_config']['heuristic']['waste_reduction']

    dummy_state = np.array([
        [
            [-2.0, -2.0],
            [0.1,  0.0],
            [0.5,  0.0],
            [0.2,  0.1],
            [0.3,  0.5],
            [0.1,  0.4],
            [0.9,  0.4],
        ]
    ], dtype='float32')
    
    node_sample_size = 3
    
    solver = WasteReductionHeuristic(node_sample_size, heuristic_opts)

    solver.solve(dummy_state)

    solver.solution
