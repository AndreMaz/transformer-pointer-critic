import sys

import numpy as np
sys.path.append('.')
import json
import random

from typing import List, Tuple

from environment.custom.resource_v3.heuristic.base_heuristic import BaseHeuristic
from environment.custom.resource_v3.node import Node
from environment.custom.resource_v3.resource import Resource
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
        
        node_list = self.parse_nodes(state)
        EOS_NODE = node_list.pop(0)

        resource_list = self.parse_resources(state)
        
        while len(resource_list) > 0:
            # Randomly pick a resource
            resource_index = random.randrange(len(resource_list))
            resource = resource_list.pop(resource_index)

            copy_list = [ ] + node_list

            self.place_single_resource(resource, copy_list, EOS_NODE)
            
        # Store a reference with the solution
        self.solution = [EOS_NODE] + node_list
    
    def place_single_resource(self, resource, node_list, EOS_NODE):
        
        # Now do the fit first
        allocated = False
        while len(node_list) > 0:
            # Randomly pick a node
            node_index = random.randrange(len(node_list))
            node: Node = node_list.pop(node_index)

            if node.can_fit_resource(resource):
                node.insert_req(resource)
                allocated = True
                break

        if not allocated:
            # Place at EOS node
            EOS_NODE.insert_req(resource)

if __name__ == "__main__":
    with open(f"configs/ResourceV3.json") as json_file:
        params = json.load(json_file)

    heuristic_opts = params['tester_config']['heuristic']['random']

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
    
    solver = RandomHeuristic(node_sample_size, heuristic_opts)

    solver.solve(dummy_state)