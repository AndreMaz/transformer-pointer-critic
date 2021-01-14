import sys

import numpy as np
sys.path.append('.')
import json

from typing import List, Tuple

from environment.custom.resource_v3.env import ResourceEnvironmentV3
from environment.custom.resource_v3.node import Node
from environment.custom.resource_v3.resource import Resource
from operator import itemgetter, attrgetter

class GreedyHeuristic():
    def __init__(self,
                env: ResourceEnvironmentV3,
                opts: dict
                ):
        super(GreedyHeuristic, self).__init__()

        self.resource_batch_id = 0
        self.env = env

        self.node_list = self.parse_nodes()

        self.EOS_NODE = self.node_list[0]

    def reset(self):
        self.resource_batch_id = 0
        
        self.node_list = []

    def parse_nodes(self) -> List[Node]:

        state, _, _, _  = self.env.state()
        
        batch_size = state.shape[0]

        assert batch_size == 1, 'Heuristic only works for problems with batch size equal to 1!'

        nodes = state[0, :self.env.node_sample_size, :]
        # resources = state[:, env.bin_sample_size:, :]
        
        node_list = []
        for id, node in enumerate(nodes):
            node_list.append(
                Node(
                    0, # Nodes are created in first batch, i.e., state from env
                    id,
                    node[0],
                    node[1],
                    node[2],
                )
            )
        
        return node_list

    def parse_resources(self, state):
        batch_size = state.shape[0]

        assert batch_size == 1, 'Heuristic only works for problems with batch size equal to 1!'

        resources = state[0, self.env.node_sample_size:, :]

        resource_list = []
        for id, resource in enumerate(resources):
            resource_list.append(
                Resource(
                    self.resource_batch_id,
                    id,
                    resource[0],
                    resource[1],
                    resource[2],
                )
            )

        self.resource_batch_id += 1
        
        return resource_list


    def solve(self, state):
        
        resource_list = self.parse_resources(state)
        # Sort the reqs in a descending order
        resource_list: List[Resource] = sorted(resource_list, key=resource_sorting_fn, reverse=True)
        
        for resource in resource_list:
            
            # Compute dominant resource of each node and current request
            diffs = []
            for node in self.node_list:
                diffs.append( 
                    (compute_dominant_resource(node, resource), node)
                )

            # Sort the nodes by dominant resource
            sorted_nodes: Tuple[float, Node] = sorted(diffs, key=node_sorting_fn, reverse=True)

            # First fit
            diff, selected_node = sorted_nodes[0]
            if (diff > 0):
                selected_node.insert_req(resource)
            else:
                self.EOS_NODE.insert_req(resource)
    
        return
    
    def print_info(self, elem_list: list):
        for elem in elem_list:
            elem.print()
            # print(elem.get_stats())

    def print_node_stats(self, print_details = False):
        for node in self.node_list:
            node.print(print_details)

def compute_dominant_resource(node: Node, resource: Resource):
        diff_cpu = node.remaining_CPU - resource.CPU
        diff_ram = node.remaining_RAM - resource.RAM
        diff_mem = node.remaining_MEM - resource.MEM

        return min(diff_cpu, diff_ram, diff_mem)


def node_sorting_fn(e: Tuple[float, Node]):
    return e[0]
    # return (node.remaining_CPU, node.remaining_RAM, node.remaining_MEM)

def resource_sorting_fn(elem: Resource):
    return (
        max(elem.CPU, elem.RAM, elem.MEM),
        ( elem.CPU + elem.RAM + elem.MEM )/3
    )

if __name__ == "__main__":
    env_name = 'Resource'

    with open(f"configs/ResourceV2.json") as json_file:
        params = json.load(json_file)

    env_config = params['env_config']

    env_config['batch_size'] = 1

    env = ResourceEnvironmentV2(env_name, env_config)

    heuristic_type = params['tester_config']['heuristic']['type']
    heuristic_opts = params['tester_config']['heuristic'][f'{heuristic_type}']

    dummy_state = np.array([
        [
            [1, 2, 3],
            [5, 2, 6],
            [2, 1, 4],
            [3, 5, 8],
        ]
    ], dtype='float32')
    # Dummy data
    env.batch = dummy_state
    env.node_sample_size = 2
    env.batch_size = 1
    
    state, _, _, _ = env.state()


    solver = GreedyHeuristic(env, heuristic_opts)

    solver.solve(state)