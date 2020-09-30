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

        ###########################################
        ##### PROBLEM CONFIGS FROM JSON FILE ######
        ###########################################

        self.batch_size: int = opts['batch_size']
        self.num_features: int = opts['num_features']
        self.num_resources: int = opts['num_resources']
        
        self.EOS_CODE: int = opts['EOS_CODE']

        self.resource_sample_size: int = opts['resource_sample_size']
        assert self.num_resources >= self.resource_sample_size, 'Resource sample size should be less than total number of resources'

        self.num_bins: int = opts['num_bins'] + 1 # + 1 because of the EOS bin 
        self.bin_sample_size: int = opts['bin_sample_size'] + 1 # + 1 because of the EOS bin
        assert self.num_bins >= self.bin_sample_size, 'Bins sample size should be less than total number of bins'

        self.normalization_factor: int = opts['normalization_factor']

        self.num_user_levels: int = opts['num_user_levels']

        self.num_task_types: int = opts['num_task_types']
    
        self.min_resource_CPU: int = opts['min_resource_CPU']
        self.max_resource_CPU: int = opts['max_resource_CPU']
        self.min_resource_RAM: int = opts['min_resource_RAM']
        self.max_resource_RAM: int = opts['max_resource_RAM']
        self.min_resource_MEM: int = opts['min_resource_MEM']
        self.max_resource_MEM: int = opts['max_resource_MEM']

        self.min_bin_CPU: int = opts['min_bin_CPU']
        self.max_bin_CPU: int = opts['max_bin_CPU']
        self.min_bin_RAM: int = opts['min_bin_RAM']
        self.max_bin_RAM: int = opts['max_bin_RAM']
        self.min_bin_MEM: int = opts['min_bin_MEM']
        self.max_bin_MEM: int = opts['max_bin_MEM']

        self.min_bin_range_type: int = opts['min_bin_range_type']
        self.max_bin_range_type: int = opts['max_bin_range_type']

        ################################################
        ##### MATERIALIZED VARIABLES FROM CONFIGS ######
        ################################################

        self.tasks = list(range(0, self.num_task_types))

        self.EOS_BIN = np.full((1, self.num_features), self.EOS_CODE, dtype='float32')
        
        self.total_bins, self.total_resources = self.generate_dataset()

        # Generate the IDs of the resources and bins
        self.binIDS = list(range(1, self.num_bins)) # Skip the 0 because it will be allways the EOS bin
        self.resourceIDS = list(range(0, self.num_resources))

        # Problem batch
        self.batch, self.history = self.generate_batch()
        # Default masks
        # Will be updated during at each step() call
        self.bin_net_mask,\
            self.resource_net_mask,\
            self.mha_used_mask = self.generate_masks()

    def reset(self):
        return

    def state(self):
        return

    def step(self, backpack_ids: list, item_ids: list):
        return
    
    def generate_dataset(self):
        bins = np.zeros((self.num_bins, self.num_features), dtype='float32')

        # Set the first EOS bin
        bins[0] = self.EOS_BIN
        for i in range(1, self.num_bins):
            # Available CPU, RAM and MEM resources
            bins[i, 0] = randint(
                self.min_bin_CPU,
                self.max_bin_CPU
            ) / self.normalization_factor
            
            bins[i, 1] = randint(
                self.min_bin_RAM,
                self.max_bin_RAM
            ) / self.normalization_factor

            bins[i, 2] = randint(
                self.min_bin_MEM,
                self.max_bin_MEM
            ) / self.normalization_factor

            # Range of tasks that node can process without any penalty
            num_tasks_for_bin = randint(self.min_bin_range_type, self.max_bin_range_type)

            task_lower_index = randint(
                0,
                self.num_task_types - ( 1 + num_tasks_for_bin)
            )

            bins[i, 3] = self.tasks[task_lower_index]
            bins[i, 4] = self.tasks[task_lower_index + num_tasks_for_bin]
        

        resources = np.zeros((self.num_resources, self.num_features), dtype='float32')

        for i in range(self.num_resources):
            resources[i, 0] = randint(
                self.min_resource_CPU,
                self.max_resource_CPU
            ) / self.normalization_factor

            resources[i, 1] = randint(
                self.min_resource_RAM,
                self.max_resource_RAM
            ) / self.normalization_factor

            resources[i, 2] = randint(
                self.min_resource_MEM,
                self.max_resource_MEM
            ) / self.normalization_factor

            resources[i, 3] = randint(0, self.num_task_types - 1)
            
            # User type will be generated on-the-fly
            resources[i, 4] = -1

        return bins, resources


    def generate_batch(self):
        history = [1]

        elem_size = self.bin_sample_size + self.resource_sample_size

        batch = np.zeros((self.batch_size, elem_size, self.num_features), dtype='float32')

        for batch_id in range(self.batch_size):
            
            # Set the EOS bin/node
            batch[batch_id, 0] = self.EOS_BIN

            # Shuffle the bins and select a sample
            np.random.shuffle(self.binIDS)
            bins_sample_ids = self.binIDS[:self.bin_sample_size - 1]

            for i in range(1, self.bin_sample_size):
                # Pop the ID
                id = bins_sample_ids.pop(0)
                # Get the bin by ID
                bin = self.total_bins[id]

                # Set the bin/node
                batch[batch_id, i, :] = bin

            # Shuffle the resources and select a sample
            np.random.shuffle(self.resourceIDS)
            resources_sample_ids = self.resourceIDS[:self.resource_sample_size]

            start = self.bin_sample_size
            end = self.bin_sample_size + self.resource_sample_size
            for i in range(start, end):
                # Pop the ID
                id = resources_sample_ids.pop(0)
                # Get the resource by ID
                resource  = self.total_resources[id]
                batch[batch_id, i, :] = resource
                
                batch[batch_id, i, 4] = randint(0, self.num_user_levels)

        return batch, history

    def generate_masks(self):
        elem_size = self.bin_sample_size + self.resource_sample_size

        # Represents positions marked as "0" where resource Ptr Net can point
        resource_net_mask = np.zeros((self.batch_size, elem_size), dtype='float32')
        # Represents positions marked as "0" where bin Ptr Net can point
        bin_net_mask = np.ones(
            (self.batch_size, elem_size), dtype='float32')

        # Default mask for resources
        for batch_id in range(self.batch_size):
            for i in range(self.bin_sample_size):
                resource_net_mask[batch_id, i] = 1

        # Default mask for bin
        bin_net_mask = bin_net_mask - resource_net_mask

        # For Transformer's multi head attention
        mha_used_mask = np.zeros_like(resource_net_mask)
        mha_used_mask = mha_used_mask[:, np.newaxis, np.newaxis, :]

        return bin_net_mask, resource_net_mask, mha_used_mask
        
    def print_history(self):
        return

    def add_stats_to_agent_config(self, agent_config: dict):
        agent_config['num_resources'] = self.resource_sample_size
        agent_config['num_bins'] = self.bin_sample_size

        agent_config['tensor_size'] = self.bin_sample_size + self.resource_sample_size
        
        agent_config['batch_size'] = self.batch_size

        agent_config['vocab_size'] = len(self.total_bins) + len(self.total_resources)
    
        return agent_config

    def build_feasible_mask(self, state, items, backpack_net_mask):
        return

    
if __name__ == "__main__":
    env_name = 'Knapsack'

    with open(f"configs/Resource.json") as json_file:
        params = json.load(json_file)

    env_config = params['env_config']

    env = Resource(env_name, env_config)
