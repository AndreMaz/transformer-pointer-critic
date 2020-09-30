import sys

sys.path.append('.')

import json
import numpy as np
import tensorflow as tf
from random import randint

from environment.base.base import BaseEnvironment

class Resource(BaseEnvironment):
    def __init__(self, name: str, opts: dict):
        super(Resource, self).__init__(name)

        self.load_from_file = opts['load_from_file']
        self.location = opts['location']

        
        self.batch_size: int = opts['batch_size']
        self.num_items: int = opts['num_items']
        self.item_sample_size: int = opts['item_sample_size']
        
        assert self.num_items >= self.item_sample_size, 'Item sample size should be less than total number of items'

        self.num_backpacks: int = opts['num_backpacks'] + 1 # + 1 because of the EOS backpack
        self.backpack_sample_size: int = opts['backpack_sample_size'] + 1 # + 1 because of the EOS backpack

        assert self.num_backpacks >= self.backpack_sample_size, 'Backpacks sample size should be less than total number of backpacks'

        self.EOS_CODE: int = opts['EOS_CODE']

        self.normalization_factor: int = opts['normalization_factor']

        self.min_item_value: int = opts['min_item_value']
        self.max_item_value: int = opts['max_item_value']
        self.min_item_weight: int = opts['min_item_weight']
        self.max_item_weight: int = opts['max_item_weight']

        self.min_backpack_capacity: int = opts['min_backpack_capacity']
        self.max_backpack_capacity: int = opts['max_backpack_capacity']

        self.EOS_BACKPACK = np.array((self.EOS_CODE, self.EOS_CODE), dtype='float32')

        if self.load_from_file:
            self.total_backpacks, self.total_items = self.load_problem()
        else:
            self.total_backpacks, self.total_items = self.generate_dataset()

        # Generate the IDs of the items and backpacks
        self.backpackIDS = list(range(1, self.num_backpacks)) # Skip the 0 because it will be allways the EOS backpack
        self.itemIDS = list(range(0, self.num_items))

        # Problem batch
        self.batch, self.history = self.generate_batch()
        # Default masks
        # Will be updated during at each step() call
        self.backpack_net_mask,\
            self.item_net_mask,\
            self.mha_used_mask = self.generate_masks()

    def reset(self):
        return

    def state(self):
        return

    def step(self, backpack_ids: list, item_ids: list):
        return

    def generate_batch(self):
        return

    def generate_masks(self):
        return

    def print_history(self):
        return

    def add_stats_to_agent_config(self, agent_config: dict):
        agent_config['num_items'] = self.num_items
        agent_config['num_backpacks'] = self.backpack_sample_size
        agent_config['tensor_size'] = self.backpack_sample_size + self.item_sample_size
        agent_config['num_items'] = self.item_sample_size
        agent_config['batch_size'] = self.batch_size

        agent_config['vocab_size'] = len(self.total_backpacks) + len(self.total_items)
    
        return agent_config

    def build_feasible_mask(self, state, items, backpack_net_mask):
        return

    
if __name__ == "__main__":
    env_name = 'Knapsack'

    with open(f"configs/Resource.json") as json_file:
        params = json.load(json_file)

    env_config = params['env_config']

    env = Resource(env_name, env_config)
