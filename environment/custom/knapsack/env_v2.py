import sys

sys.path.append('.')

import json
import numpy as np
import tensorflow as tf
from random import randint

from environment.base.base import BaseEnvironment
from environment.custom.knapsack.backpack import Backpack as History

class KnapsackV2(BaseEnvironment):
    def __init__(self, name: str, opts: dict):
        super(KnapsackV2, self).__init__(name)

        self.load_from_file = opts['load_from_file']
        self.location = opts['location']

        
        self.batch_size: int = opts['batch_size']
        self.num_resources: int = opts['num_resources']
        self.resource_sample_size: int = opts['resource_sample_size']
        
        assert self.num_resources >= self.resource_sample_size, 'Item sample size should be less than total number of resources'

        self.num_bins: int = opts['num_bins'] + 1 # + 1 because of the EOS bin
        self.bin_sample_size: int = opts['bin_sample_size'] + 1 # + 1 because of the EOS bin

        assert self.num_bins >= self.bin_sample_size, 'Backpacks sample size should be less than total number of bins'

        self.EOS_CODE: int = opts['EOS_CODE']

        self.normalization_factor: int = opts['normalization_factor']

        self.min_resource_value: int = opts['min_resource_value']
        self.max_resource_value: int = opts['max_resource_value']
        self.min_resource_weight: int = opts['min_resource_weight']
        self.max_resource_weight: int = opts['max_resource_weight']

        self.min_bin_capacity: int = opts['min_bin_capacity']
        self.max_bin_capacity: int = opts['max_bin_capacity']

        self.EOS_BACKPACK = np.array((self.EOS_CODE, self.EOS_CODE), dtype='float32')

        if self.load_from_file:
            self.total_bins, self.total_resources = self.load_problem()
        else:
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
        self.batch, self.history = self.generate_batch()
        self.bin_net_mask,\
            self.resource_net_mask,\
            self.mha_used_mask = self.generate_masks()

        return self.state()

    def state(self):
        return self.batch.copy(),\
            self.bin_net_mask.copy(),\
            self.resource_net_mask.copy(),\
            self.mha_used_mask.copy()

    def step(self, bin_ids: list, resource_ids: list, feasible_bin_mask):
        # rewards = []
        rewards = np.zeros((self.batch_size, 1), dtype="float32")

        # Default is not done
        isDone = False

        # Default mask for resources
        for batch_id in range(self.batch_size):
            bin_id = bin_ids[batch_id]
            resource_id = resource_ids[batch_id]

            bin = self.batch[batch_id, bin_id]
            resource = self.batch[batch_id, resource_id]

            resource_weight = resource[0]
            resource_value = resource[1]

            bin_capacity = bin[0]
            bin_load = bin[1]

            # Update the bin entry
            if (bin_id != 0):                
                assert bin_load + resource_weight <= bin_capacity ,\
                f'Backpack {bin_id} is overloaded. Maximum capacity: {bin_capacity} || Current load: {bin_load} || Item Weight: {resource_weight}'

                self.batch[batch_id, bin_id, 1] = bin_load + resource_weight

            # Add to history
            history_entry: History = self.history[batch_id][bin_id]
            history_entry.add_resource(resource_id, resource_weight, resource_value)

            # Update the masks
            # Item taken mask it
            self.resource_net_mask[batch_id, resource_id] = 1
            self.mha_used_mask[batch_id, :, :, resource_id] = 1

            # Mask the bin if it's full
            if (bin_capacity == bin_load + resource_weight):
                self.bin_net_mask[batch_id, bin_id] = 1
                self.mha_used_mask[batch_id, :, :, bin_id] = 1

            if (bin_id == 0):
                reward = 0 # No reward. Placed at EOS bin
            else:
                reward = resource_value

            rewards[batch_id][0] = reward

        info = {
             'bin_net_mask': self.bin_net_mask.copy(),
             'resource_net_mask': self.resource_net_mask.copy(),
             'mha_used_mask': self.mha_used_mask.copy(),
             'num_resource_to_place': self.num_resources
        }

        if np.all(self.resource_net_mask == 1):
            isDone = True
        
        return self.batch.copy(), tf.convert_to_tensor(rewards), isDone, info

    def generate_dataset(self):
        # Num bins + 1 for EOS
        bins = np.zeros((self.num_bins, 2), dtype='float32')
        
        # Skip the first EOS bin
        bins[0] = self.EOS_BACKPACK
        for i in range(1, self.num_bins):
            bins[i, 0] = randint(
                self.min_bin_capacity,
                self.max_bin_capacity
            ) / self.normalization_factor
            
            bins[i, 1] = 0 # Current load

        resources = np.zeros((self.num_resources, 2), dtype='float32')

        for i in range(self.num_resources):
            resources[i, 0] = randint(
                self.min_resource_weight,
                self.max_resource_weight
            ) / self.normalization_factor

            resources[i, 1] = randint(
                self.min_resource_value,
                self.max_resource_value
            ) / self.normalization_factor

        return bins, resources

    def generate_batch(self):
        history = [] # For info and stats

        elem_size = self.bin_sample_size + self.resource_sample_size

        # Init empty batch
        batch = np.zeros((self.batch_size, elem_size, 2), dtype='float32')

        for batch_id in range(self.batch_size):
            problem = []

            # Set the EOS bin
            batch[batch_id, 0] = self.EOS_BACKPACK
            
            problem.append(History(0, 0)) # EOS bin is always empty

            # Shuffle the bins and select a sample
            np.random.shuffle(self.binIDS)
            bins_sample_ids = self.binIDS[:self.bin_sample_size - 1]

            for i in range(1, self.bin_sample_size):
                # Pop the ID
                id = bins_sample_ids.pop(0)
                # Get the bin by ID
                bin = self.total_bins[id]

                problem.append(History(i, bin[0]))

                batch[batch_id, i, 0] = bin[0] # Set total capacity
                batch[batch_id, i, 1] = bin[1] # Set current load = 0

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
                batch[batch_id, i, 0] = resource[0] # Set weight
                batch[batch_id, i, 1] = resource[1] # Set value

            history.append(problem)

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
        for batch_id in range(self.batch_size):
            total_packed_resources = 0
            total_packed_value = 0
            total_packed_weight = 0
            print('_________________________________')
            for bp in self.history[batch_id]:
                bp.print()
                total_packed_resources += len(bp.resources)
                total_packed_value += bp.current_value
                total_packed_weight += bp.current_load
            print(f'\nTotal Packed Items:  {total_packed_resources} || Total Packed Value {total_packed_value} || Total Packed Weight {total_packed_weight}')
            print('_________________________________')
    
    def add_stats_to_agent_config(self, agent_config: dict):
        agent_config['num_resources'] = self.resource_sample_size
        agent_config['num_bins'] = self.bin_sample_size

        agent_config['tensor_size'] = self.bin_sample_size + self.resource_sample_size
        
        agent_config['batch_size'] = self.batch_size

        agent_config['vocab_size'] = len(self.total_bins) + len(self.total_resources)
    
        return agent_config

    def build_feasible_mask(self, state, resources, bin_net_mask):
        
        batch = state.shape[0]

        resource_net_mask = np.ones_like(bin_net_mask)
        resource_net_mask -= bin_net_mask

        # Extract weights
        # Reshape into (batch, 1)
        resource_weight = np.reshape(resources[:, 0], (batch, 1))

        bin_capacity = state[:, :, 0]
        bin_current_load = state[:, :, 1]

        totals = bin_capacity - (bin_current_load + resource_weight)
        # EOS is always available for poiting
        totals[:,0] = 0
        # Can't point to resources positions
        totals *= resource_net_mask

        binary_masks = tf.cast(
            tf.math.less(totals, 0), tf.float32
        )

        # Merge the masks
        mask = tf.maximum(binary_masks, bin_net_mask)

        return tf.cast(mask, dtype="float32").numpy()

    def convert_to_ortools_input(self, problem_id = 0):

        assert problem_id < self.batch_size, f'Problem ID is out of bounds. Must be less than {self.batch_size}'

        # Select by ID from the batch
        problem = self.batch[problem_id]
        p_bps = problem[:self.bin_sample_size]
        p_resources = problem[self.bin_sample_size:]

        data = {}

        weights = []
        values = []
        for resource in p_resources:
            weights.append(int(resource[0]))
            values.append(int(resource[1]))

        data['weights'] = weights
        data['values'] = values
        data['resources'] = list(range(len(weights)))
        data['num_resources'] = len(weights)

        bins = []
        for bin in p_bps:
            bins.append(int(bin[0]))
        
        data['bin_capacities'] = bins
        data['bins'] = list(range(len(bins)))

        return data
    
    def save_problem(self):
        bins = {}
        for i, bin in enumerate(self.total_bins):
            bins[f'{i}'] = {}
            bins[f'{i}']['capacity'] = float(bin[0])

        resources = {}
        for i, resource in enumerate(self.total_resources):
            resources[f'{i}'] = {}
            resources[f'{i}']['weight'] = float(resource[0])
            resources[f'{i}']['value'] = float(resource[1])

        problem = {
            "bins": bins,
            "resources": resources
        }

        with open(self.location, 'w') as fp:
            json.dump(problem, fp, indent=4)

    def load_problem(self):
        with open(self.location) as json_file:
            problem = json.load(json_file)

        bins_dict: dict = problem['bins']
        self.num_bins = len(bins_dict)
        bins = np.zeros((self.num_bins, 2), dtype='float32')

        for id, bin in enumerate(bins_dict.values()):
            bins[id, 0] = bin['capacity']
            
            # EOS has the save value
            if (id == 0): 
                bins[id, 1] = bin['capacity']
            else:
                bins[id, 1] = 0

        resources_dict: dict = problem['resources']
        self.num_resources = len(resources_dict)
        resources = np.zeros((self.num_resources, 2), dtype='float32')

        for id, resource in enumerate(resources_dict.values()):
            resources[id, 0] = resource['weight']
            resources[id, 1] = resource['value']

        return bins, resources


    def validate_history(self):
        for problem in self.history:
            for bin in problem:
                if bin.is_valid() == False:
                    return False

        return True

if  __name__ == "__main__": # pragma: no cover
    env_name = 'Knapsack'

    with open(f"configs/KnapsackV2.json") as json_file:
        params = json.load(json_file)

    env_config = params['env_config']

    env = KnapsackV2(env_name, env_config)

    env.generate_dataset()
    env.generate_batch()
    env.generate_masks()

    # bin_ids = [0 , 1]
    # resource_ids = [3, 4]

    # next_step, rewards, isDone, info = env.step(bin_ids, resource_ids)
    # env.print_history()

    # state, bin_net_mask, resource_net_mask = env.reset()

    # resources = state[[0,1], [3,4]]
    # resources[0][0] = 999
    
    state = np.array(
        [[[-2, -2],
          [14,  0],
          [39,  1],
          [19, 60],
          [ 1, 13]]],
          dtype='float32' 
    )
    
    resources = np.array(
        [[19, 60]],
        dtype='float32'
        )
    resources_mask = np.array(
        [[1., 1., 1., 0., 1.]],
        dtype='float32'
        )
    bp_mask = np.array(
        [[0., 0., 0., 1., 1.]],
        dtype='float32'
    )

    new_mask = env.build_feasible_mask(state, resources, bp_mask)
    print(new_mask)

    env.convert_to_ortools_input()
