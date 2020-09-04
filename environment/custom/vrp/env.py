import sys

sys.path.append('.')

from math import ceil, sqrt
import numpy as np
import tensorflow as tf


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
        self.visited_nodes = 0

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
    
    def reset(self):
        self.visited_nodes = 0

        self.batch, self.history = self.generate_batch()

        self.backpack_net_mask,\
            self.item_net_mask,\
            self.mha_used_mask = self.generate_masks()

        return self.state()

    def state(self):
        return self.batch.copy(),\
            self.backpack_net_mask.copy(),\
            self.item_net_mask.copy(),\
            self.mha_used_mask.copy()

    def step(self, vehicle_ids: list, node_ids: list):
        # rewards = []
        rewards = np.zeros((self.batch_size, 1), dtype="float32")

        # Default is not done
        isDone = False

        # Default mask for items
        for batch_id in range(self.batch_size):
            vehicle_id = vehicle_ids[batch_id]
            node_id = node_ids[batch_id]

            vehicle = self.batch[batch_id, vehicle_id]
            node = self.batch[batch_id, node_id]

            node_x = node[0]
            node_y = node[1]
            node_demand = node[2]

            vehicle_x = vehicle[0]
            vehicle_y = vehicle[1]
            vehicle_current_capacity = vehicle[2]
            
            xd = vehicle_x - node_x
            yd = vehicle_y - node_y

            distance = round(sqrt(xd*xd + yd*yd))

            history_entry: History = self.history[batch_id][str(vehicle_id)]
            history_entry.add_node(node_id, node_x, node_y, node_demand)

            assert vehicle_current_capacity - node_demand >= 0, \
                f'Vehicle {vehicle_id} is overloaded. Available Capacity {vehicle_current_capacity} ||  Item Weight: {node_demand}'

            # Update vehicle location
            self.batch[batch_id, vehicle_id, 0] = node_x
            self.batch[batch_id, vehicle_id, 1] = node_y
            self.batch[batch_id, vehicle_id, 2] = vehicle_current_capacity - node_demand

            # Update the masks
            # Node visited taken mask it
            if node_demand != 0:
                self.item_net_mask[batch_id, node_id] = 1
                self.mha_used_mask[batch_id, :, :, node_id] = 1

            # Mask the vehicle if it's full
            if vehicle_current_capacity - node_demand == 0:
                self.backpack_net_mask[batch_id, vehicle_id] = 1
                self.mha_used_mask[batch_id, :, :, vehicle_id] = 1

            rewards[batch_id][0] = -1 * distance
        
        self.visited_nodes += 1
        
        # Visited all nodes
        # Time to return all vehicles to the depot
        if self.node_sample_size - 1 == self.visited_nodes:
            self.item_net_mask[:, 0] = 0

        if self.visited_nodes == self.node_sample_size + self.vehicle_sample_size:
        # if np.all(self.item_net_mask == 1):
            isDone = True

        info = {
             'backpack_net_mask': self.backpack_net_mask.copy(),
             'item_net_mask': self.item_net_mask.copy(),
             'mha_used_mask': self.mha_used_mask.copy(),
             'num_items_to_place': self.node_sample_size
        }

        return self.batch.copy(), rewards, isDone, info

    def generate_batch(self):
        history = [] # For info and stats

        elem_size = self.node_sample_size + self.vehicle_sample_size
        
        # Init empty batch
        batch = np.zeros((self.batch_size, elem_size, self.num_features), dtype='float32')

        for batch_id in range(self.batch_size):
            problem = {}

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

                problem[str(i)]= History(
                    i,
                    vehicle[0],
                    vehicle[1],
                    vehicle[2]
                )
                
                batch[batch_id, i, 0] = vehicle[0] # Set the X coord
                batch[batch_id, i, 1] = vehicle[1] # Set the Y coord
                batch[batch_id, i, 2] = vehicle[2] # Remaining capacity

            history.append(problem)

        return batch, history

    def generate_masks(self):
        elem_size = self.node_sample_size + self.vehicle_sample_size
        
        # When selecting an node you can't point to the vehicle
        # Represents positions marked as "0" where item Ptr Net can point
        nodes_net_mask = np.zeros((self.batch_size, elem_size), dtype='float32')
        
        # Represents positions marked as "0" where backpack Ptr Net can point
        vehicles_net_mask = np.ones(
            (self.batch_size, elem_size), dtype='float32')

        for batch_id in range(self.batch_size):
            for i in range(self.node_sample_size):
                nodes_net_mask[batch_id, i] = 1

        vehicles_net_mask = vehicles_net_mask - nodes_net_mask
        
        # We can't point the depot
        # Only allow when all nodes are visited
        vehicles_net_mask[:, 0] = 1

        mha_used_mask = np.zeros_like(nodes_net_mask)
        mha_used_mask = mha_used_mask[:, np.newaxis, np.newaxis, :]

        return nodes_net_mask, vehicles_net_mask, mha_used_mask

    def add_stats_to_agent_config(self, agent_config: dict):
        agent_config['num_items'] = self.node_sample_size + self.vehicle_sample_size
        agent_config['num_backpacks'] = self.vehicle_sample_size
        agent_config['tensor_size'] = self.node_sample_size + self.vehicle_sample_size

        agent_config['batch_size'] = self.batch_size

        agent_config['vocab_size'] = len(self.total_vehicles) + len(self.total_nodes)
    
        return agent_config

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

        item_net_mask = np.ones_like(backpack_net_mask)
        item_net_mask -= backpack_net_mask

        # Extract weights
        # Reshape into (batch, 1)
        item_weight = np.reshape(items[:, 2], (batch, 1))

        # backpack_capacity = state[:, :, 0]
        backpack_current_load = state[:, :, 2]

        totals = backpack_current_load - item_weight
        # EOS is always available for poiting
        # totals[:,0] = 0
        # Can't point to items positions
        totals *= item_net_mask

        binary_masks = tf.cast(
            tf.math.less(totals, 0), tf.float32
        )

        # Merge the masks
        mask = tf.maximum(binary_masks, backpack_net_mask)

        # for i in range(batch):
        #    if np.all(mask[i] == 1):
        #        print(mask[i])
        
        return tf.cast(mask, dtype="float32")

    def print_history(self):
        for batch_id in range(self.batch_size):
            print('_________________________________')
            for bp in self.history[batch_id].values():
                bp.print()
            print('_________________________________')

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