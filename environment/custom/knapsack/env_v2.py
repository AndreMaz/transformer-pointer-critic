import sys

from numpy.core.numeric import inf
sys.path.append('.')

import json
import numpy as np
from random import randint

from environment.base.base import BaseEnvironment
from environment.custom.knapsack.item import Item
from environment.custom.knapsack.backpack import Backpack, EOS_BACKPACK, NORMAL_BACKPACK

class KnapsackV2(BaseEnvironment):
    def __init__(self, name: str, opts: dict):
        super(KnapsackV2, self).__init__(name)

        self.load_from_file = opts['load_from_file']
        self.location = opts['location']

        
        self.batch_size: int = opts['batch_size']
        self.num_items: int = opts['num_items']
        self.item_sample_size = opts['item_sample_size']
        
        assert self.num_items > self.item_sample_size, 'Item sample size should be less than total number of items'

        self.num_backpacks: int = opts['num_backpacks']

        self.EOS_CODE: int = opts['EOS_CODE']

        self.normalization_factor: int = opts['normalization_factor']

        self.min_item_value: int = opts['min_item_value']
        self.max_item_value: int = opts['max_item_value']
        self.min_item_weight: int = opts['min_item_weight']
        self.max_item_weight: int = opts['max_item_weight']

        self.min_backpack_capacity: int = opts['min_backpack_capacity']
        self.max_backpack_capacity: int = opts['max_backpack_capacity']
        
        self.EOS_BACKPACK = np.array((self.EOS_CODE, self.EOS_CODE), dtype='float16')
        self.backpackIDS = list(range(0, self.num_backpacks))
        self.itemIDS = list(range(0, self.num_items))

        self.total_backpacks, self.total_items = self.generate_dataset()
        self.batch = self.generate_batch()
        # Default masks
        # Will be updated during step
        self.backpack_net_mask, self.item_net_mask = self.generate_masks()

    def reset(self):
        self.batch = self.generate_batch()
        self.backpack_net_mask, self.item_net_mask = self.generate_masks()

    def state(self):

        return self.batch.copy(),\
            self.backpack_net_mask.copy(),\
            self.item_net_mask.copy()

    def step(self, backpack_ids: list, item_ids: list):
        rewards = []

        # Default is not done
        isDone = False

        # Default mask for items
        for batch_id in range(self.batch_size):
            backpack_id = backpack_ids[batch_id]
            item_id = item_ids[batch_id]

            backpack = self.batch[batch_id, backpack_id]
            item = self.batch[batch_id, item_id]

            item_weight = item[0]
            item_value = item[1]

            backpack_capacity = backpack[0]
            backpack_load = backpack[1]

            # Update the backpack entry
            if (backpack_id != 0):
                assert backpack_capacity > backpack_load + item_weight,\
                f'Batch {batch_id}: Backpack {backpack_id} is overloaded'

                self.batch[batch_id, backpack_id, 1] = backpack_load + item_weight

            # Update the masks
            # Item taken mask it
            self.item_net_mask[batch_id, item_id] = 1
            # Mask the backpack if it's full
            if (backpack_capacity == backpack_load + item_weight):
                self.backpack_net_mask[batch_id, backpack_id] = 1

            if (backpack_id == 0):
                reward = 0 # No reward. Placed at EOS backpack
            else:
                reward = item_value

            rewards.append(reward)

        info = {
             'backpack_net_mask': self.backpack_net_mask.copy(),
             'item_net_mask': self.item_net_mask.copy(),
             'num_items_to_place': self.num_items
        }

        if np.all(self.item_net_mask == 1):
            isDone = True
        
        return self.batch.copy(), rewards, isDone, info

    def generate_dataset(self):
        # Num backpacks + 1 for EOS
        backpacks = np.zeros((self.num_backpacks, 2), dtype='float16')
        
        # Skip the first EOS backpack
        for i in range(self.num_backpacks):
            backpacks[i, 0] = randint(
                self.min_backpack_capacity,
                self.max_backpack_capacity
            ) / self.normalization_factor
            
            backpacks[i, 1] = 0 # Current load

        items = np.zeros((self.num_items, 2), dtype='float16')

        for i in range(self.num_items):
            items[i, 0] = randint(
                self.min_item_weight,
                self.max_item_weight
            ) / self.normalization_factor

            items[i, 1] = randint(
                self.min_item_value,
                self.max_item_value
            ) / self.normalization_factor

        return backpacks, items

    def generate_batch(self):
        # + 1 for the EOS backpack
        elem_size = 1 + self.num_backpacks + self.item_sample_size

        # Init empty batch
        batch = np.zeros((self.batch_size, elem_size, 2), dtype='float16')

        for batch_id in range(self.batch_size):
            # Set the EOS backpack
            batch[batch_id, 0] = self.EOS_BACKPACK

            for i in range(1, self.num_backpacks + 1):
                batch[batch_id, i, 0] = self.total_backpacks[i-1][0] # Set total capacity
                batch[batch_id, i, 1] = self.total_backpacks[i-1][1] # Set current load = 0

            # Shuffle the items and select a sample
            np.random.shuffle(self.itemIDS)
            items_sample_ids = self.itemIDS[:self.item_sample_size]

            start = self.num_backpacks + 1
            end = self.num_backpacks + 1 + self.item_sample_size
            for i in range(start, end):
                # Pop the ID
                id = items_sample_ids.pop(0)
                # Get the item by ID
                item  = self.total_items[id]
                batch[batch_id, i, 0] = item[0] # Set weight
                batch[batch_id, i, 1] = item[1] # Set value

        return batch

    def generate_masks(self):
        # + 1 for the EOS backpack
        elem_size = 1 + self.num_backpacks + self.item_sample_size

        # Represents positions marked as "0" where item Ptr Net can point
        item_net_mask = np.zeros((self.batch_size, elem_size), dtype='float16')
        # Represents positions marked as "0" where backpack Ptr Net can point
        backpack_net_mask = np.ones(
            (self.batch_size, elem_size), dtype='float16')

        # Default mask for items
        for batch_id in range(self.batch_size):
            for i in range(self.num_backpacks + 1):
                item_net_mask[batch_id, i] = 1

        # Default mask for backpack
        backpack_net_mask = backpack_net_mask - item_net_mask

        return backpack_net_mask, item_net_mask

if __name__ == "__main__":
    env_name = 'Knapsack'

    with open(f"configs/Knapsack.json") as json_file:
        params = json.load(json_file)

    env_config = params['env_config']

    env = KnapsackV2(env_name, env_config)

    env.generate_dataset()
    env.generate_batch()
    env.generate_masks()

    backpack_ids = [0 , 1]
    item_ids = [3, 4]

    next_step, rewards, isDone, info = env.step(backpack_ids, item_ids)

    print(info)
