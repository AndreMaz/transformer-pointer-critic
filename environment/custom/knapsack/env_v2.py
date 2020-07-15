import json
import numpy as np

from environment.base.base import BaseEnvironment
from environment.custom.knapsack.item import Item
from environment.custom.knapsack.backpack import Backpack, EOS_BACKPACK, NORMAL_BACKPACK

class KnapsackV2(BaseEnvironment):
    def __init__(self, name: str, opts: dict):
        super(KnapsackV2, self).__init__(name)

        self.load_from_file = opts['load_from_file']
        self.location = opts['location']

        self.batch_dim = opts['batch_size']

        self.normalization_factor = opts['normalization_factor']

        self.new_items_on_reset = opts['new_items_on_reset']
        self.num_items = opts['num_items']
        self.num_backpacks = opts['num_backpacks']

        self.sampling_mode = opts['sampling_mode']
        self.sample_size = opts['sample_size']

        self.min_item_value = opts['min_item_value']
        self.max_item_value = opts['max_item_value']
        self.min_item_weight = opts['min_item_weight']
        self.max_item_weight = opts['max_item_weight']

        self.min_backpack_capacity = opts['min_backpack_capacity']
        self.max_backpack_capacity = opts['max_backpack_capacity']

        # Represents the number of backpacks (including EOS backpack) and items
        self.tensor_size = (self.num_backpacks + 1) + (self.num_items)

        #if self.load_from_file:
            # self.backpacks, self.items = self.load_problem()
        #else:
        self.backpacks, self.items = self.generate()

        self.backpack_ids, self.item_ids = self.generate_positional_encodings()

    def generate(self):

        items = {}
        backpacks = {}

        # ID that's used for indexation of elements
        GLOBAL_ID = 0
        # Create EOS Backpack
        backpacks[f'{GLOBAL_ID}'] = Backpack(
            GLOBAL_ID,
            self.normalization_factor,
            EOS_BACKPACK,
            min_capacity = 0,
            max_capacity = 0
        )
        GLOBAL_ID += 1
        
        # Create Normal Backpacks
        for _ in range(self.num_backpacks):
            backpack = Backpack(
                GLOBAL_ID,
                self.normalization_factor,
                NORMAL_BACKPACK,
                min_capacity = self.min_backpack_capacity,
                max_capacity = self.max_backpack_capacity
            )

            GLOBAL_ID += 1
            backpacks[f'{backpack.id}'] = backpack

        for _ in range(self.num_items):
            item = Item(
                GLOBAL_ID,
                self.normalization_factor,
                min_value = self.min_item_value,
                max_value = self.max_item_value,
                min_weight = self.min_item_weight,
                max_weight = self.max_item_weight
            )

            GLOBAL_ID += 1
            items[f'{item.id}'] = item
            
        return backpacks, items

    def generate_positional_encodings(self):
        eos_id = self.backpacks['0'].id

        backpacks_ids = []
        for backpack in self.backpacks.values():
            if (backpack.type == NORMAL_BACKPACK):
                backpacks_ids.append(backpack.id)
        
        item_ids = []
        for item in self.items.values():
            item_ids.append(item.id)

        return [eos_id] + backpacks_ids, item_ids

    def shuffle_all(self):
        self.shuffle_backpacks()
        self.shuffle_items()

    def shuffle_backpacks(self):
        eos_id = self.backpack_ids[0]
        rest_ids = list(self.backpack_ids[1:])
        np.random.shuffle(rest_ids)

        self.backpack_ids = [eos_id] + rest_ids
    
    def shuffle_items(self):
        np.random.shuffle(self.item_ids)
    
    def sample_items(self):
        sample_ids = []

        if self.sample_size <= len(self.item_ids):
            for _ in range(self.sample_size):
                sample_ids.append(
                    self.item_ids.pop(0) # pop from front
                )
        else:
            while len(self.item_ids) != 0:
                sample_ids.append(
                    self.item_ids.pop(0) # pop from front
                )

        return sample_ids

    def step(self, backpack, item):
        """Takes one step, i.e., placed an item in the backpack

        Args:
            backpack ([type]): ID of the backpack
            item ([type]): ID of the item 

        Returns:
            [type]: new state
        """

        backpack_id = str(backpack)
        item_id = str(item)

        # Default value
        isDone = False

        backpack: Backpack = self.backpacks[backpack_id]
        item: Item = self.items[item_id]
    
        is_valid, capacity_diff, backpack_value = backpack.add_item(item)

        if backpack.type == NORMAL_BACKPACK:
            if (is_valid == False):
                # Give negative rewards because the backpack is overloaded
                reward = -1 * item.value
                isDone = True
            else:
                reward = item.value
        else:
            reward = 0
            isDone = False

        # Backpack's capacities not exceeded
        # Check if all item were taken
        if is_valid:
            isDone = self.all_items_taken()

    
        # Compute the masks for the Ptr Nets
        backpack_net_mask, item_net_mask = self.compute_masks()    
    
        info = {
             'backpack_net_mask': backpack_net_mask,
             'item_net_mask': item_net_mask,
             'num_items_to_place': self.num_items
        }
        next_state = self.convert_to_tensor()
        
        return next_state, reward, isDone, info

    def compute_masks(self, sample_backpack_ids, sample_item_ids):
        sample_backpack_ids = list(sample_backpack_ids)
        sample_item_ids = list(sample_item_ids)

        num_items = len(sample_item_ids)
        num_backpacks = len(sample_backpack_ids)
        tensor_size = num_backpacks + num_items
        
        # Represents positions marked as "0" where item Ptr Net can point
        item_net_mask = np.zeros((self.batch_dim, tensor_size), dtype='float16')
        # Represents positions marked as "0" where backpack Ptr Net can point
        backpack_net_mask = np.ones((self.batch_dim, tensor_size), dtype='float16')

        # Default mask. Items can't point to backpacks positions
        for index, ids in enumerate(sample_backpack_ids):
            item_net_mask[0, index] = 1

        # Backpacks mask is the reverse
        backpack_net_mask = backpack_net_mask - item_net_mask

        # Iterate over the backpacks mask the ones that are full
        for mask_position in range(num_backpacks):
            id = sample_backpack_ids.pop(0)
            backpack = self.backpacks[str(id)]
            if backpack.is_full():
                backpack_net_mask[0, mask_position] = 1

        # Iterate over the items that were already taken and mask them
        for mask_position in range(num_backpacks, tensor_size):
            id = sample_item_ids.pop(0)
            item = self.items[str(id)]
            if item.is_taken():
                item_net_mask[0, mask_position] = 1

        return backpack_net_mask, item_net_mask

    def convert_to_tensor(self, sample_backpack_ids, sample_item_ids):
        sample_backpack_ids = list(sample_backpack_ids)
        sample_item_ids = list(sample_item_ids)

        num_items = len(sample_item_ids)
        num_backpacks = len(sample_backpack_ids)
        tensor_size = num_backpacks + num_items


        tensor_env = np.zeros((self.batch_dim, tensor_size, 2), dtype='float32')

        for mask_position in range(num_backpacks):
            id = sample_backpack_ids.pop(0)
            backpack = self.backpacks[str(id)]
            if (backpack.type != 'eos'):
                tensor_env[0, mask_position, 0] = backpack.capacity
                tensor_env[0, mask_position, 1] = backpack.current_capacity

        for mask_position in range(num_backpacks, tensor_size):
            id = sample_item_ids.pop(0)
            item = self.items[str(id)]
            tensor_env[0, mask_position, 0] = item.weight
            tensor_env[0, mask_position, 1] = item.value
        
        return tensor_env
    
    def add_stats_to_agent_config(self, agent_config):
        agent_config['num_items'] = self.num_items
        agent_config['num_backpacks'] = self.num_backpacks
        agent_config['tensor_size'] = self.tensor_size
    
        return agent_config

    def print_stats(self, verbose = False):
        print(f'Batch Size: {self.batch_dim}')
        print(f'Number of Items: {self.num_items}')
        print(f'Number of Backpacks: {self.num_backpacks}')

        if verbose:
            print(f'All Items:')
            for item in self.items.values():
                item.print_stats()

            print(f'Backpacks')
            for backpack in self.backpacks.values():
                backpack.print_stats()
                print('_________________________________')