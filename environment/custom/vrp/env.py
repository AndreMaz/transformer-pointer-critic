import sys

sys.path.append('.')

from math import ceil
import numpy as np


from environment.base.base import BaseEnvironment
from environment.custom.vrp.datasets.parser import load_problem, DEPOT
from environment.custom.vrp.vehicle import Vehicle as History

class CVRP(BaseEnvironment):
    def __init__(self, name: str, opts: dict):
        super(CVRP, self).__init__(name)
        
        self.dir_path = opts['dir_path']
        self.prob_name = opts['problem_name']

        self.batch_size = opts['batch_size']
        self.node_sample_size = opts['node_sample_size']
        self.vehicle_sample_size = opts['vehicle_sample_size']

        self.num_features = opts['num_features']

        self.parsed_prob, self.parsed_sol = load_problem(self.dir_path, self.prob_name)
        self.total_demand = self.parsed_prob['total_demand']
        self.vehicle_capacity = self.parsed_prob['capacity']
        self.num_vehicles = ceil(self.total_demand / self.vehicle_capacity)
        
        self.num_nodes = self.parsed_prob['dimension']

        self.total_vehicles, self.total_nodes, self.EOS_DEPOT  = self.generate_dataset()
        
        self.vehiclesIDs = list(range(0, self.num_vehicles))
        self.nodeIDs = list(range(1, self.num_nodes)) # Skip the 0 because it will be allways the EOS depot 

        self.batch, self.history = self.generate_batch()

        self.backpack_net_mask,\
            self.item_net_mask,\
            self.mha_used_mask = self.generate_masks()

    def generate_batch(self):
        history = [] # For info and stats

        elem_size = self.node_sample_size + self.vehicle_sample_size
        
        # Init empty batch
        batch = np.zeros((self.batch_size, elem_size, self.num_features), dtype='float32')

        for batch_id in range(self.batch_size):
            problem = []

            batch[batch_id, 0] = self.EOS_DEPOT

            np.random.shuffle(self.nodeIDs)
            node_sample_ids = self.nodeIDs[:self.node_sample_size]

            for i in range(1, self.node_sample_size):
                # Pop the ID
                id = node_sample_ids.pop(0)
                # Get the backpack ID
                node = self.total_nodes[id]

                batch[batch_id, i, 0] = node[0] # Set the X coord
                batch[batch_id, i, 1] = node[1] # Set the Y coord
                batch[batch_id, i, 2] = node[2] # Set the demand
            
            np.random.shuffle(self.vehiclesIDs)
            vehicle_sample_ids = self.vehiclesIDs[:self.node_sample_size]

            start = self.node_sample_size
            end = self.node_sample_size + self.vehicle_sample_size
            
            for i in range(start, end):
                # Pop the ID
                id = vehicle_sample_ids.pop(0)

                vehicle = self.total_vehicles[id]

                problem.append(History(
                    i,
                    vehicle[0],
                    vehicle[1],
                    vehicle[2]
                ))

                batch[batch_id, i, 0] = vehicle[0] # Set the X coord
                batch[batch_id, i, 1] = vehicle[1] # Set the Y coord
                batch[batch_id, i, 2] = vehicle[2] # Remaining capacity

            history.append(problem)

        return batch, history

    def generate_masks(self):
        elem_size = self.node_sample_size + self.vehicle_sample_size

        # Represents positions marked as "0" where item Ptr Net can point
        nodes_net_mask = np.zeros((self.batch_size, elem_size), dtype='float32')
        # Represents positions marked as "0" where backpack Ptr Net can point
        vehicles_net_mask = np.ones(
            (self.batch_size, elem_size), dtype='float32')

        for batch_id in range(self.batch_size):
            for i in range(self.vehicle_sample_size):
                nodes_net_mask[batch_id, i] = 1

        vehicles_net_mask = vehicles_net_mask - nodes_net_mask

        mha_used_mask = np.zeros_like(nodes_net_mask)
        mha_used_mask = mha_used_mask[:, np.newaxis, np.newaxis, :]

        return vehicles_net_mask, nodes_net_mask, mha_used_mask

    def generate_dataset(self):
        nodes = np.zeros((self.num_nodes, self.num_features), dtype='float32')
        prob_nodes = self.parsed_prob['nodes']
        depot = None

        for i, node in enumerate(prob_nodes):
            if node['type'] == DEPOT:
                depot = node

            nodes[i, 0] = node['x']
            nodes[i, 1] = node['y']
            nodes[i, 2] = node['demand']

        vehicles = np.zeros((self.num_vehicles, self.num_features), dtype='float32')
        # In the beggining all vehicles are at the depot
        for i in range(self.num_vehicles):
            vehicles[i, 0] = depot['x']
            vehicles[i, 1] = depot['y']
            vehicles[i, 2] = self.vehicle_capacity

        EOS_DEPOT = np.zeros((1, self.num_features), dtype='float32')
        EOS_DEPOT[0,0] = depot['x']
        EOS_DEPOT[0,1] = depot['y']
        EOS_DEPOT[0,2] = depot['demand']

        return vehicles, nodes, EOS_DEPOT
    
    def build_feasible_mask(self, state, items, backpack_net_mask):
        
        batch = state.shape[0]

if __name__ == "__main__":
    env_name = 'CVRP'

    env_config = {
        "description": "Environment configs.",

        "load_from_file": False,
        "dir_path": "./environment/custom/vrp/datasets",
        "problem_name": "A-n65-k9",

        "num_features": 3,

        "batch_size": 32,
        
        "node_sample_size": 5,
        "vehicle_sample_size": 5,
    }

    env = CVRP(env_name, env_config)
    

    state = [[ 25.,  51.,   0.],
            [ 55.,  25.,  10.],
            [ 75.,  83.,  10.],
            [ 47.,  19.,   7.],
            [ 93.,  75.,  24.],
            [ 25.,  51., 100.],
            [ 25.,  51., 100.],
            [ 25.,  51., 100.],
            [ 25.,  51., 100.],
            [ 25.,  51., 100.]]
    
    # node = 