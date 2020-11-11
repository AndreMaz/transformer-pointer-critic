import sys

from tensorflow.python.ops.gen_batch_ops import batch
sys.path.append('.')
import json

from environment.custom.resource.env import ResourceEnvironment
from environment.custom.resource.node import Node
from environment.custom.resource.resource import Resource

class GreedyHeuristic():
    def __init__(self,
                env: ResourceEnvironment,
                opts: dict
                ):
        super(GreedyHeuristic, self).__init__()

        self.resource_batch_id = 0
        self.env = env

        self.penalizer = self.env.penalizer

        self.task_normalization_factor = self.env.task_normalization_factor

        # self.denormalize_input: bool = opts['denormalize']

        self.node_list = self.parse_nodes()

    def reset(self):
        self.resource_batch_id = 0
        
        self.node_list = []

    def parse_nodes(self):

        state, _, _, _  = self.env.state()
        
        batch_size = state.shape[0]

        assert batch_size == 1, 'Heuristic only works for problems with batch size equal to 1!'

        nodes = state[0, :env.bin_sample_size, :]
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
                    self.task_normalization_factor
                )
            )
        
        return node_list

    def parse_resources(self, state):
        batch_size = state.shape[0]

        assert batch_size == 1, 'Heuristic only works for problems with batch size equal to 1!'

        resources = state[0, env.bin_sample_size:, :]

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

        # Sort the resources

        # For each resource
        # Find appropriate nodes
        # Sort them

        # If none found
        # Sort all the nodes and use the first-fit approach

        return

if __name__ == "__main__":
    env_name = 'Resource'

    with open(f"configs/Resource.json") as json_file:
        params = json.load(json_file)

    env_config = params['env_config']

    env_config['batch_size'] = 1

    env = ResourceEnvironment(env_name, env_config)

    env.state()

    heuristic_type = params['tester_config']['heuristic']['type']
    heuristic_opts = params['tester_config']['heuristic'][f'{heuristic_type}']

    solver = GreedyHeuristic(env, heuristic_opts)

    state, _, _, _ = env.state()

    solver.solve(state)