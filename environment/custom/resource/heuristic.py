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

        self.env = env
        self.opts = opts

    def solver(self, env: ResourceEnvironment, opts: dict):
        
        nodes, resources = parse_input(env, opts)

        return 1

    def parse_input(self, env: ResourceEnvironment, opts: dict):

        state, _, _, _  = env.state()
        
        batch_size = state.shape[0]

        assert batch_size == 1, 'Heuristic only works for problems with batch size equal to 1!'

        nodes = state[:, :env.bin_sample_size, :]
        resources = state[:, env.bin_sample_size:, :]
        
        if opts['denormalize'] == True:
            nodes, resources = denormalize(nodes, resources)

        node_list = []
        for id, node in enumerate(nodes):
            node_list.append(
                Node(
                    0, # Any value
                    id,
                    node[0],
                    node[1],
                    node[2],
                    node[3],
                    node[4],
                )
            )
        
        resource_list = []
        for id, node in enumerate(resources):
            resource_list.append(
                Resource(

                )
            )
        

        return 1, 1


    def parse_resource_list(self, resources):
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

    solver(env, heuristic_opts)