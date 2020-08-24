import sys

sys.path.append('.')

from math import ceil
import numpy as np

from environment.base.base import BaseEnvironment
from environment.custom.vrp.datasets.parser import load_problem, DEPOT

class CVRP(BaseEnvironment):
    def __init__(self, name: str, opts: dict):
        super(CVRP, self).__init__(name)
        
        self.dir_path = opts['dir_path']
        self.prob_name = opts['problem_name']

        self.parsed_prob, self.parsed_sol = load_problem(self.dir_path, self.prob_name)
        self.total_demand = self.parsed_prob['total_demand']
        self.vehicle_capacity = self.parsed_prob['capacity']
        self.num_vehicles = ceil(self.total_demand / self.vehicle_capacity)
        
        self.num_nodes = self.parsed_prob['dimension']

        self.total_vechiles, self.total_nodes = self.generate_dataset()
        
        self.vehiclesIDs = list(range(0, self.num_vehicles))
        self.nodeIDs = list(range(0, self.num_nodes))

        self.batch, self.history = self.generate_batch()

    def generate_batch(self):
        return 1

    def generate_dataset(self):
        nodes = np.zeros((self.num_nodes, 3), dtype='float32')
        prob_nodes = self.parsed_prob['nodes']
        depot = None

        for i, node in enumerate(prob_nodes):
            if node['type'] == DEPOT:
                depot = node

            nodes[i, 0] = node['x']
            nodes[i, 1] = node['y']
            nodes[i, 2] = node['demand']

        vehicles = np.zeros((self.num_vehicles, 3), dtype='float32')
        # In the beggining all vehicles are at the depot
        for i in range(self.num_vehicles):
            vehicles[i, 0] = depot['x']
            vehicles[i, 1] = depot['y']
            vehicles[i, 2] = self.vehicle_capacity

        return vehicles, nodes

if __name__ == "__main__":
    env_name = 'CVRP'

    env_config = {
        "dir_path": './environment/custom/vrp/datasets',
        "problem_name": "A-n65-k9"
    }

    env = CVRP(env_name, env_config)
    print(len(env.prob))