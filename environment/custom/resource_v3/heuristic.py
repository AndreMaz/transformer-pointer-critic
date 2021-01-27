import sys

import numpy as np
sys.path.append('.')
import json

from typing import List, Tuple

from environment.custom.resource_v3.node import Node
from environment.custom.resource_v3.resource import Resource
from operator import itemgetter, attrgetter

class GreedyHeuristic():
    def __init__(self,
                num_nodes: int,
                opts: dict
                ):
        super(GreedyHeuristic, self).__init__()

        self.resource_batch_id = 0
        self.num_nodes = num_nodes

        self.solution = []

    def reset(self):
        self.resource_batch_id = 0
        
        self.solution = []

    def parse_nodes(self, state) -> List[Node]:

        batch_size = state.shape[0]

        assert batch_size == 1, 'Heuristic only works for problems with batch size equal to 1!'

        nodes = state[0, :self.num_nodes, :]
        # resources = state[:, env.bin_sample_size:, :]
        
        node_list = []
        for id, node in enumerate(nodes):
            node_list.append(
                Node(
                    0, # Nodes are created in first batch, i.e., state from env
                    id,
                    node
                )
            )
        
        return node_list

    def parse_resources(self, state):
        batch_size = state.shape[0]

        assert batch_size == 1, 'Heuristic only works for problems with batch size equal to 1!'

        resources = state[0, self.num_nodes:, :]

        resource_list = []
        for id, resource in enumerate(resources):
            resource_list.append(
                Resource(
                    self.resource_batch_id,
                    id,
                    resource
                )
            )

        self.resource_batch_id += 1
        
        return resource_list


    def solve(self, state):
        
        node_list = self.parse_nodes(state)
        EOS_NODE = node_list.pop(0)

        resource_list = self.parse_resources(state)
        
        # Sort the resources in a descending order
        resource_list: List[Resource] = sorted(resource_list, key=resource_sorting_fn, reverse=True)
        
        for resource in resource_list:
            self.place_single_resource(resource, node_list, EOS_NODE)
            

        # Store a reference with the solution
        self.solution = [EOS_NODE] + node_list
    
    def place_single_resource(self, resource, node_list, EOS_NODE):
        
        diffs = compute_potential_placement_diffs(resource, node_list)

        # Sort the nodes by dominant resource
        sorted_nodes: Tuple[float, Node] = sorted(diffs, key=node_sorting_fn, reverse=True)

        # First fit
        diff, selected_node = sorted_nodes[0]
        if (diff > 0):
            selected_node.insert_req(resource)
        else:
            # Place at EOS node
            EOS_NODE.insert_req(resource)

    def print_info(self, elem_list: list):
        for elem in elem_list:
            elem.print()
            # print(elem.get_stats())

    def print_node_stats(self, print_details = False):
        for node in self.solution:
            node.print(print_details)

def compute_potential_placement_diffs(resource, node_list) -> Tuple[float, Node]:
        # Compute dominant resource of each node and current request
        diffs = []
        for node in node_list:
            diffs.append( 
                (compute_dominant_resource(node, resource), node)
            )

        return diffs

def compute_dominant_resource(node: Node, resource: Resource):
        diff_cpu = node.remaining_CPU - resource.CPU
        diff_ram = node.remaining_RAM - resource.RAM
        diff_mem = node.remaining_MEM - resource.MEM

        return min(diff_cpu, diff_ram, diff_mem)

def node_sorting_fn(e: Tuple[float, Node]):
    return e[0]
    # return (node.remaining_CPU, node.remaining_RAM, node.remaining_MEM)

def resource_sorting_fn(elem: Resource):
    return max(elem.CPU, elem.RAM, elem.MEM)
    
if __name__ == "__main__":
    env_name = 'Resource'

    with open(f"configs/ResourceV3.json") as json_file:
        params = json.load(json_file)

    heuristic_type = params['tester_config']['heuristic']['type']
    heuristic_opts = params['tester_config']['heuristic'][f'{heuristic_type}']

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
    
    solver = GreedyHeuristic(node_sample_size, heuristic_opts)

    solver.solve(dummy_state)