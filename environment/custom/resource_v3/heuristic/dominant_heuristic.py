import sys

import numpy as np
sys.path.append('.')
import json

from typing import List, Tuple

from environment.custom.resource_v3.heuristic.base_heuristic import BaseHeuristic
from environment.custom.resource_v3.node import Node
from environment.custom.resource_v3.resource import Resource
from operator import itemgetter, attrgetter

class DominantResourceHeuristic(BaseHeuristic):
    def __init__(self,
                num_nodes: int,
                normalization_factor: int,
                opts: dict
                ):
        super(DominantResourceHeuristic, self).__init__(num_nodes, normalization_factor)

        self.resource_sort_descending = opts['resource_sort_descending']
        self.node_sort_descending = opts['node_sort_descending']

        self.generate_name()
    
    def generate_name(self):
        self.name = f'dominant_resource_ASC_{self.resource_sort_descending}_node_ASC_{self.node_sort_descending}'

    def solve(self, state):
        
        node_list = self.parse_nodes(state)
        EOS_NODE = node_list.pop(0)

        resource_list = self.parse_resources(state)
        
        # Sort the resources in a descending order
        resource_list: List[Resource] = sorted(
            resource_list,
            key=resource_sorting_fn,
            reverse=self.resource_sort_descending
        )
        
        for resource in resource_list:
            self.place_single_resource(resource, node_list, EOS_NODE)
            
        # Store a reference with the solution
        self.solution = [EOS_NODE] + node_list
    
    def place_single_resource(self, resource: Resource, node_list: List[Node], EOS_NODE: Node):
        
        diffs = compute_potential_placement_diffs(resource, node_list)

        # Sort the nodes by dominant resource
        sorted_nodes: Tuple[float, Node] = sorted(
            diffs,
            key=node_sorting_fn,
            reverse=self.node_sort_descending
        )

        # Now do the fit first
        allocated = False
        for diff, node in sorted_nodes:
            if (diff >= 0):
                node.insert_req(resource)
                allocated = True
                break

        if not allocated:
            # Place at EOS node
            EOS_NODE.insert_req(resource)

def compute_potential_placement_diffs(resource, node_list) -> Tuple[float, Node]:
        # Compute dominant resource of each node and current request
        diffs = []
        for node in node_list:
            diffs.append( 
                (node.compute_dominant_resource(resource), node)
            )

        return diffs

def node_sorting_fn(e: Tuple[float, Node]):
    return e[0]
    # return (node.remaining_CPU, node.remaining_RAM, node.remaining_MEM)

def resource_sorting_fn(elem: Resource):
    return max(elem.CPU, elem.RAM, elem.MEM)
    
if  __name__ == "__main__": # pragma: no cover
    with open(f"configs/ResourceV3.json") as json_file:
        params = json.load(json_file)

    heuristic_opts = params['tester_config']['heuristic']['dominant_resource']

    dummy_state = np.array([
        [
            [-2, -2, -2],
            [ 1,  2,  3],
            [ 5,  2,  6],
            [ 2,  1,  4],
            [ 3,  5,  8],
        ]
    ], dtype='float32')
    
    node_sample_size = 3
    
    solver = DominantResourceHeuristic(node_sample_size, heuristic_opts)

    solver.solve(dummy_state)