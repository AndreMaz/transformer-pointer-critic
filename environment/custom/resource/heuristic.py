import sys

from tensorflow.python.ops.gen_batch_ops import batch
sys.path.append('.')
import json

from typing import List, Tuple

from environment.custom.resource.env import ResourceEnvironment
from environment.custom.resource.node import Node
from environment.custom.resource.resource import Resource
from operator import itemgetter, attrgetter

class GreedyHeuristic():
    def __init__(self,
                env: ResourceEnvironment,
                opts: dict
                ):
        super(GreedyHeuristic, self).__init__()

        self.EOS_CODE = env.EOS_CODE

        self.resource_batch_id = 0
        self.env = env

        self.penalizer = self.env.penalizer

        self.task_normalization_factor = self.env.task_normalization_factor

        # self.denormalize_input: bool = opts['denormalize']

        self.node_list = self.parse_nodes()

        self.EOS_NODE = self.node_list[0]

    def reset(self):
        self.resource_batch_id = 0
        
        self.node_list = []

    def parse_nodes(self) -> List[Node]:

        state, _, _, _  = self.env.state()
        
        batch_size = state.shape[0]

        assert batch_size == 1, 'Heuristic only works for problems with batch size equal to 1!'

        nodes = state[0, :self.env.bin_sample_size, :]
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
                    node[3],
                    node[4],
                    self.penalizer,
                    self.task_normalization_factor,
                    gather_stats=True
                )
            )
        
        return node_list

    def parse_resources(self, state):
        batch_size = state.shape[0]

        assert batch_size == 1, 'Heuristic only works for problems with batch size equal to 1!'

        resources = state[0, self.env.bin_sample_size:, :]

        resource_list = []
        for id, resource in enumerate(resources):
            resource_list.append(
                Resource(
                    self.resource_batch_id,
                    id,
                    resource[0],
                    resource[1],
                    resource[2],
                    resource[3],
                    resource[4],
                )
            )

        self.resource_batch_id += 1
        
        return resource_list


    def solve(self, state):
        
        resource_list = self.parse_resources(state)
        # resource_list: List[Resource] = sorted(resource_list, key=attrgetter("request_type"), reverse=True)
        resource_list: List[Resource] = sorted(resource_list, key=resource_sorting_fn, reverse=True)
        
        # Sort the resources
        for resource in resource_list:
            # Look for nodes that have cached info
            matching_nodes = self.find_matching_nodes(resource)

            matching_nodes: List[Node] = sorted(matching_nodes, key=node_sorting_fn, reverse=True)
            complete_list: List[Node] = sorted(self.node_list, key=node_sorting_fn, reverse=True)

            # Concate matching nodes and then the complete list
            candidate_nodes = matching_nodes + complete_list

            # Now do the fit first
            allocated = False
            for node in candidate_nodes:
                if node.lower_task != self.EOS_CODE and node.upper_task != self.EOS_CODE:
                    isValid, _, _, _ = node.validate(resource)

                    if isValid:
                        allocated = True
                        node.insert_resource(resource)
                        break
            
            if not allocated:
                self.EOS_NODE.insert_resource(resource)
            
            allocated = False

        # self.print_info(self.node_list)
        
        return
    
    def find_matching_nodes(self, resource: Resource):
        matching_nodes = []

        for node in self.node_list:
            if node.lower_task != self.EOS_CODE and node.upper_task != self.EOS_CODE:
                if node.lower_task <= resource.task <= node.upper_task:
                    matching_nodes.append(node)

        return matching_nodes
    
    def print_info(self, elem_list: list):
        for elem in elem_list:
            elem.print()
            # print(elem.get_stats())

    def print_node_stats(self, print_details = False):
        for node in self.node_list:
            node.print(print_details)

def node_sorting_fn(node: Node):
    return (node.remaining_CPU, node.remaining_RAM, node.remaining_MEM)

def resource_sorting_fn(elem: Resource):
    return (elem.request_type, elem.CPU, elem.RAM, elem.MEM)

if __name__ == "__main__":
    env_name = 'Resource'

    with open(f"configs/Resource.json") as json_file:
        params = json.load(json_file)

    env_config = params['env_config']

    env_config['batch_size'] = 1

    env = ResourceEnvironment(env_name, env_config)

    # env.state()

    heuristic_type = params['tester_config']['heuristic']['type']
    heuristic_opts = params['tester_config']['heuristic'][f'{heuristic_type}']

    solver = GreedyHeuristic(env, heuristic_opts)

    state, _, _, _ = env.state()

    solver.solve(state)