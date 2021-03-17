
from typing import List, Tuple
from environment.custom.resource_v3.node import Node
from environment.custom.resource_v3.resource import Resource

class BaseHeuristic():
    def __init__(self,
                num_nodes:int
                ):
        self.name = 'base_heuristic'

        self.resource_batch_id = 0
        self.num_nodes = num_nodes

        self.solution = []

    def generate_name(self, state):
        raise NotImplementedError('Method "generate_name" not implemented')

    def solve(self, state):
        raise NotImplementedError('Method "solve" not implemented')

    def place_single_resource(self, resource, node_list, EOS_NODE):
        raise NotImplementedError('Method "place_single_resource" not implemented')

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
    

    def print_info(self, elem_list: list): # pragma: no cover
        for elem in elem_list:
            elem.print()
            # print(elem.get_stats())

    def print_node_stats(self, print_details = False): # pragma: no cover
        for node in self.solution:
            node.print(print_details)
