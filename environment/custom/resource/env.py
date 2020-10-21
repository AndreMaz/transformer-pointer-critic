import sys
from typing import List

sys.path.append('.')

import json
import numpy as np
import tensorflow as tf
from random import randint, randrange

from environment.base.base import BaseEnvironment
from environment.custom.resource.node import Node as History
from environment.custom.resource.penalty import Penalty
from environment.custom.resource.reward import Reward

class ResourceEnvironment(BaseEnvironment):
    def __init__(self, name: str, opts: dict):
        super(ResourceEnvironment, self).__init__(name)

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

        self.resource_normalization_factor: int = opts['resource_normalization_factor']
        self.task_normalization_factor: int = opts['task_normalization_factor']

        self.num_user_levels: int = opts['num_user_levels']
        self.reward_per_level: List[int] = opts['reward_per_level']

        assert self.num_user_levels + 1 == len(self.reward_per_level), 'Length of reward_per_level must be equal to the num_user_levels'

        self.misplace_reward_penalty: int = opts['misplace_reward_penalty']

        self.num_task_types: int = opts['num_task_types']

        self.CPU_misplace_penalty: int = opts['CPU_misplace_penalty'] / self.resource_normalization_factor
        self.RAM_misplace_penalty: int = opts['RAM_misplace_penalty'] / self.resource_normalization_factor
        self.MEM_misplace_penalty: int = opts['MEM_misplace_penalty'] / self.resource_normalization_factor
        
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
        
        # Class responsible for computing penalties for each placement
        self.penalizer = Penalty(
            self.CPU_misplace_penalty,
            self.RAM_misplace_penalty,
            self.MEM_misplace_penalty
        )

        # Class responsible form computing rewards for each placement
        self.rewarder = Reward(
            self.reward_per_level,
            self.misplace_reward_penalty,
            self.penalizer
        )

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

    def step(self, bin_ids: list, resource_ids: list):
        rewards = np.zeros((self.batch_size, 1), dtype="float32")

        # Default is not done
        isDone = False

        # Default mask for resources
        for batch_id in range(self.batch_size):
            bin_id = bin_ids[batch_id]
            resource_id = resource_ids[batch_id]

            bin = self.batch[batch_id, bin_id]
            resource = self.batch[batch_id, resource_id]

            node: History = self.history[batch_id][bin_id]

            # Add resource to bin
            remaining_resources = node.add_resource(
                resource_id,
                resource[0],
                resource[1],
                resource[2],
                resource[3],
                resource[4],
            )

            # No reward. Placed at EOS bin
            if bin_id == 0:
                reward = 0
            else:
                # Compute reward
                reward = self.rewarder.compute_reward(
                    self.batch[batch_id],
                    self.bin_sample_size,
                    bin,
                    resource,
                )
            
            rewards[batch_id][0] = reward

            # Update the remaining node resources
            if (bin_id != 0):
                assert np.all(self.batch[batch_id, bin_id, :3] > 0), f'Batch {batch_id}: Node {bin_id} is overloaded!'
                self.batch[batch_id, bin_id, :3] = remaining_resources

            # Update the masks
            # Item taken mask it
            self.resource_net_mask[batch_id, resource_id] = 1
            self.mha_used_mask[batch_id, :, :, resource_id] = 1

            if (np.all(self.batch[batch_id, bin_id, :3] == 0)):
                self.bin_net_mask[batch_id, bin_id] = 1

        info = {
             'bin_net_mask': self.bin_net_mask.copy(),
             'resource_net_mask': self.resource_net_mask.copy(),
             'mha_used_mask': self.mha_used_mask.copy(),
             'num_resource_to_place': self.num_resources
        }

        if np.all(self.resource_net_mask == 1):
            isDone = True
        
        return self.batch.copy(), rewards, isDone, info
    
    def generate_dataset(self):
        bins = np.zeros((self.num_bins, self.num_features), dtype='float32')

        # Set the first EOS bin
        bins[0] = self.EOS_BIN
        for i in range(1, self.num_bins):
            # Available CPU, RAM and MEM resources
            bins[i, 0] = randint(
                self.min_bin_CPU,
                self.max_bin_CPU
            ) / self.resource_normalization_factor
            
            bins[i, 1] = randint(
                self.min_bin_RAM,
                self.max_bin_RAM
            ) / self.resource_normalization_factor

            bins[i, 2] = randint(
                self.min_bin_MEM,
                self.max_bin_MEM
            ) / self.resource_normalization_factor

            # Range of tasks that node can process without any penalty
            num_tasks_for_bin = randint(self.min_bin_range_type, self.max_bin_range_type)

            task_lower_index = randrange(
                0,
                self.num_task_types - num_tasks_for_bin
            )

            bins[i, 3] = self.tasks[task_lower_index] / self.task_normalization_factor
            bins[i, 4] = self.tasks[task_lower_index + num_tasks_for_bin] / self.task_normalization_factor
        

        resources = np.zeros((self.num_resources, self.num_features), dtype='float32')

        for i in range(self.num_resources):
            resources[i, 0] = randint(
                self.min_resource_CPU,
                self.max_resource_CPU
            ) / self.resource_normalization_factor

            resources[i, 1] = randint(
                self.min_resource_RAM,
                self.max_resource_RAM
            ) / self.resource_normalization_factor

            resources[i, 2] = randint(
                self.min_resource_MEM,
                self.max_resource_MEM
            ) / self.resource_normalization_factor

            resources[i, 3] = self.tasks[randint(0, self.num_task_types - 1)] / self.task_normalization_factor
            
            # User type will be generated on-the-fly
            resources[i, 4] = -1

        return bins, resources


    def generate_batch(self):
        history = []

        elem_size = self.bin_sample_size + self.resource_sample_size

        batch = np.zeros((self.batch_size, elem_size, self.num_features), dtype='float32')

        for batch_id in range(self.batch_size):
            problem = []
            
            # Set the EOS bin/node
            batch[batch_id, 0] = self.EOS_BIN

            problem.append(History(
                batch_id,
                0,
                self.EOS_CODE,
                self.EOS_CODE,
                self.EOS_CODE,
                self.EOS_CODE,
                self.EOS_CODE,
                self.penalizer
            ))

            # Shuffle the bins and select a sample
            np.random.shuffle(self.binIDS)
            bins_sample_ids = self.binIDS[:self.bin_sample_size - 1]

            for i in range(1, self.bin_sample_size):
                # Pop the ID
                id = bins_sample_ids.pop(0)
                # Get the bin by ID
                bin = self.total_bins[id]

                problem.append(History(
                    batch_id,
                    i,
                    bin[0], # CPU
                    bin[1], # RAM
                    bin[2], # MEM
                    bin[3], # Lower task
                    bin[4], # Upper task
                    self.penalizer
                ))

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
                
                # User type. e.g. premium or free
                batch[batch_id, i, 4] = randint(0, self.num_user_levels)

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
        
    def add_stats_to_agent_config(self, agent_config: dict):
        agent_config['num_resources'] = self.resource_sample_size
        agent_config['num_bins'] = self.bin_sample_size

        agent_config['tensor_size'] = self.bin_sample_size + self.resource_sample_size
        
        agent_config['batch_size'] = self.batch_size

        agent_config['vocab_size'] = len(self.total_bins) + len(self.total_resources)
    
        return agent_config

    def build_feasible_mask(self, state, resources, bin_net_mask):
        batch = state.shape[0]
        num_elems = state.shape[1]

        penalty_tensor = self.penalizer.tensor_representation()
        penalty_tensor = tf.tile(penalty_tensor, [num_elems, 1])
        penalty_tensor = tf.expand_dims(penalty_tensor, 0)

        resource_net_mask = np.ones_like(bin_net_mask)
        resource_net_mask -= bin_net_mask

        # Extract type of the resource
        # Reshape into (batch, 1)
        resource_type = np.reshape(resources[:, 3], (batch, 1))
        resource_demands = np.reshape(resources[:, :3], (batch, 1, 3))

        bin_lower_type = state[:, :, 3]
        bin_upper_type = state[:, :, 4]
        bin_remaining_resources = state[:,:, :3]

        # Look for places that won't be penalized
        # From: https://stackoverflow.com/a/55870586
        lower_tensor = tf.greater_equal(resource_type, bin_lower_type)
        upper_tensor = tf.less_equal(resource_type, bin_upper_type)

        in_range = tf.logical_and(lower_tensor, upper_tensor)
        # Marked as 1 = there's penalty involved
        # Marked as 0 = no penalty
        in_range = 1 - tf.cast(in_range, dtype='float32')
        in_range = tf.expand_dims(in_range, -1)
        
        resource_demands = tf.tile(resource_demands, [1, num_elems, 1])

        # Compute remaining resources after placement
        after_place = bin_remaining_resources - (resource_demands + in_range * penalty_tensor)
        
        after_place = tf.reduce_min(after_place, axis=-1)
        after_place = tf.less(after_place, 0)
        after_place = tf.cast(after_place, dtype='float32')


        # Can't point to resources positions
        after_place = tf.maximum(after_place, bin_net_mask)
        
        after_place = after_place.numpy()
        # EOS is always available for pointing
        after_place[:, 0] = 0
    
        return after_place

    def num_inserted_resources(self):
        num_inserted = 0

        for instance in self.history:
            for bin in instance:
                num_inserted += len(bin.resources)

        return num_inserted
    
    def rebuild_history(self) -> None:
        history = []

        for batch_id, instance in enumerate(self.batch):
            bins = []
            for i, bin in enumerate(instance[:self.bin_sample_size]):
                bins.append(
                    History(
                        batch_id,
                        i,
                        bin[0], # CPU
                        bin[1], # RAM
                        bin[2], # MEM
                        bin[3], # Lower task
                        bin[4], # Upper task
                        self.penalizer
                    )
                )
            
            history.append(bins)
        
        self.history = history

    def print_history(self) -> None:

        for batch_id in range(self.batch_size):
            print('_________________________________')
            for bp in self.history[batch_id]:
                bp.print()
            print('_________________________________')

    def validate_history(self):
        for problem in self.history:
            for bin in problem:
                if bin.is_valid() == False:
                    return False

        return True        

if __name__ == "__main__":
    env_name = 'Resource'

    with open(f"configs/Resource.json") as json_file:
        params = json.load(json_file)

    env_config = params['env_config']

    env = ResourceEnvironment(env_name, env_config)
    
    # bin_ids = [1 , 2]

    # resource_ids = [3, 4]

    # env.step(bin_ids, resource_ids)
    # env.rebuild_history()

    state = np.array([[
                [  0.,   0.,   0.,   0.,   0.],
                [ 10.,  20.,  30.,   0.,   2.], # Node task range [0, 2]
                [ 10.,  20.,  30.,   0.,   3.],
                [ 10.,  20.,  30.,   3.,   1.], # Resource task 15
                [ 40.,  50.,  60.,   1.,   0.]],
            [
                [   0.,    0.,    0.,   0.,   0.],
                [ 400.,  500.,  600.,   2.,   5.],
                [ 400.,  500.,  600.,   3.,   6.], # Node task range [3, 6]
                [ 100.,  200.,  300.,   0.,   1.],
                [ 400.,  500.,  600.,   6.,   1.] # Resource task 10
            ]], dtype='float32')

    resources = np.array([
            [ 10.,  20.,  30.,   3.,   1.],
            [ 400.,  500.,  600.,   6.,   1.]
        ], dtype='float32')    

    bin_net_mask = np.array([
            [0., 0., 0., 1.,  1.],
            [0., 0., 0., 1.,  1.]
        ], dtype='float32')


    env.build_feasible_mask(state, resources, bin_net_mask)
