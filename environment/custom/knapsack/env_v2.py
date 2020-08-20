import sys

sys.path.append('.')

import json
import numpy as np
import tensorflow as tf
from random import randint

from environment.base.base import BaseEnvironment
from environment.custom.knapsack.history import History
# from environment.custom.knapsack.item import Item
# from environment.custom.knapsack.backpack import Backpack, EOS_BACKPACK, NORMAL_BACKPACK

class KnapsackV2(BaseEnvironment):
    def __init__(self, name: str, opts: dict):
        super(KnapsackV2, self).__init__(name)

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

    def step(self, backpack_ids: list, item_ids: list):
        # rewards = []
        rewards = np.zeros((self.batch_size, 1), dtype="float32")

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
                assert backpack_load + item_weight <= backpack_capacity ,\
                f'Backpack {backpack_id} is overloaded. Maximum capacity: {backpack_capacity} || Current load: {backpack_load} || Item Weight: {item_weight}'

                self.batch[batch_id, backpack_id, 1] = backpack_load + item_weight

            # Add to history
            history_entry: History = self.history[batch_id][backpack_id]
            history_entry.add_item(item_id, item_weight, item_value)

            # Update the masks
            # Item taken mask it
            self.item_net_mask[batch_id, item_id] = 1
            self.mha_used_mask[batch_id, :, :, item_id] = 1

            # Mask the backpack if it's full
            if (backpack_capacity == backpack_load + item_weight):
                self.backpack_net_mask[batch_id, backpack_id] = 1
                self.mha_used_mask[batch_id, :, :, backpack_id] = 1

            if (backpack_id == 0):
                reward = 0 # No reward. Placed at EOS backpack
            else:
                reward = item_value

            rewards[batch_id][0] = reward

        info = {
             'backpack_net_mask': self.backpack_net_mask.copy(),
             'item_net_mask': self.item_net_mask.copy(),
             'mha_used_mask': self.mha_used_mask.copy(),
             'num_items_to_place': self.num_items
        }

        if np.all(self.item_net_mask == 1):
            isDone = True
        
        return self.batch.copy(), rewards, isDone, info

    def generate_dataset(self):
        # Num backpacks + 1 for EOS
        backpacks = np.zeros((self.num_backpacks, 2), dtype='float32')
        
        # Skip the first EOS backpack
        backpacks[0] = self.EOS_BACKPACK
        for i in range(1, self.num_backpacks):
            backpacks[i, 0] = randint(
                self.min_backpack_capacity,
                self.max_backpack_capacity
            ) / self.normalization_factor
            
            backpacks[i, 1] = 0 # Current load

        items = np.zeros((self.num_items, 2), dtype='float32')

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
        history = [] # For info and stats

        elem_size = self.backpack_sample_size + self.item_sample_size

        # Init empty batch
        batch = np.zeros((self.batch_size, elem_size, 2), dtype='float32')

        for batch_id in range(self.batch_size):
            problem = []

            # Set the EOS backpack
            batch[batch_id, 0] = self.EOS_BACKPACK
            
            problem.append(History(0, 0)) # EOS backpack is always empty

            # Shuffle the backpacks and select a sample
            np.random.shuffle(self.backpackIDS)
            backpacks_sample_ids = self.backpackIDS[:self.backpack_sample_size - 1]

            for i in range(1, self.backpack_sample_size):
                # Pop the ID
                id = backpacks_sample_ids.pop(0)
                # Get the backpack by ID
                backpack = self.total_backpacks[id]

                problem.append(History(i, backpack[0]))

                batch[batch_id, i, 0] = backpack[0] # Set total capacity
                batch[batch_id, i, 1] = backpack[1] # Set current load = 0

            # Shuffle the items and select a sample
            np.random.shuffle(self.itemIDS)
            items_sample_ids = self.itemIDS[:self.item_sample_size]

            start = self.backpack_sample_size
            end = self.backpack_sample_size + self.item_sample_size
            for i in range(start, end):
                # Pop the ID
                id = items_sample_ids.pop(0)
                # Get the item by ID
                item  = self.total_items[id]
                batch[batch_id, i, 0] = item[0] # Set weight
                batch[batch_id, i, 1] = item[1] # Set value

            history.append(problem)

        return batch, history

    def generate_masks(self):
        
        elem_size = self.backpack_sample_size + self.item_sample_size

        # Represents positions marked as "0" where item Ptr Net can point
        item_net_mask = np.zeros((self.batch_size, elem_size), dtype='float32')
        # Represents positions marked as "0" where backpack Ptr Net can point
        backpack_net_mask = np.ones(
            (self.batch_size, elem_size), dtype='float32')

        # Default mask for items
        for batch_id in range(self.batch_size):
            for i in range(self.backpack_sample_size):
                item_net_mask[batch_id, i] = 1

        # Default mask for backpack
        backpack_net_mask = backpack_net_mask - item_net_mask

        # For Transformer's multi head attention
        mha_used_mask = np.zeros_like(item_net_mask)
        mha_used_mask = mha_used_mask[:, np.newaxis, np.newaxis, :]

        return backpack_net_mask, item_net_mask, mha_used_mask

    def print_history(self):
        for batch_id in range(self.batch_size):
            print('_________________________________')
            for bp in self.history[batch_id]:
                bp.print()
            print('_________________________________')
    
    def add_stats_to_agent_config(self, agent_config: dict):
        agent_config['num_items'] = self.num_items
        agent_config['num_backpacks'] = self.backpack_sample_size
        agent_config['tensor_size'] = self.backpack_sample_size + self.item_sample_size
        agent_config['num_items'] = self.item_sample_size
        agent_config['batch_size'] = self.batch_size

        agent_config['vocab_size'] = len(self.total_backpacks) + len(self.total_items)
    
        return agent_config

    def build_feasible_mask(self, state, items, backpack_net_mask):
        
        batch = state.shape[0]

        item_net_mask = np.ones_like(backpack_net_mask)
        item_net_mask -= backpack_net_mask

        # Extract weights
        # Reshape into (batch, 1)
        item_weight = np.reshape(items[:, 0], (batch, 1))

        backpack_capacity = state[:, :, 0]
        backpack_current_load = state[:, :, 1]

        totals = backpack_capacity - (backpack_current_load + item_weight)
        # EOS is always available for poiting
        totals[:,0] = 0
        # Can't point to items positions
        totals *= item_net_mask

        binary_masks = tf.cast(
            tf.math.less(totals, 0), tf.float32
        )

        # Merge the masks
        mask = tf.maximum(binary_masks, backpack_net_mask)

        return tf.cast(mask, dtype="float32")

    def convert_to_ortools_input(self, problem_id = 0):

        assert problem_id < self.batch_size, f'Problem ID is out of bounds. Must be less than {self.batch_size}'

        # Select by ID from the batch
        problem = self.batch[problem_id]
        p_bps = problem[:self.backpack_sample_size]
        p_items = problem[self.backpack_sample_size:]

        data = {}

        weights = []
        values = []
        for item in p_items:
            weights.append(int(item[0]))
            values.append(int(item[1]))

        data['weights'] = weights
        data['values'] = values
        data['items'] = list(range(len(weights)))
        data['num_items'] = len(weights)

        backpacks = []
        for backpack in p_bps:
            backpacks.append(int(backpack[0]))
        
        data['bin_capacities'] = backpacks
        data['bins'] = list(range(len(backpacks)))

        return data
    
    def save_problem(self):
        backpacks = {}
        for i, backpack in enumerate(self.total_backpacks):
            backpacks[f'{i}'] = {}
            backpacks[f'{i}']['capacity'] = float(backpack[0])

        items = {}
        for i, item in enumerate(self.total_items):
            items[f'{i}'] = {}
            items[f'{i}']['weight'] = float(item[0])
            items[f'{i}']['value'] = float(item[1])

        problem = {
            "backpacks": backpacks,
            "items": items
        }

        with open(self.location, 'w') as fp:
            json.dump(problem, fp, indent=4)

    def load_problem(self):
        with open(self.location) as json_file:
            problem = json.load(json_file)

        backpacks_dict: dict = problem['backpacks']
        self.num_backpacks = len(backpacks_dict)
        backpacks = np.zeros((self.num_backpacks, 2), dtype='float32')

        for id, backpack in enumerate(backpacks_dict.values()):
            backpacks[id, 0] = backpack['capacity']
            
            # EOS has the save value
            if (id == 0): 
                backpacks[id, 1] = backpack['capacity']
            else:
                backpacks[id, 1] = 0

        items_dict: dict = problem['items']
        self.num_items = len(items_dict)
        items = np.zeros((self.num_items, 2), dtype='float32')

        for id, item in enumerate(items_dict.values()):
            items[id, 0] = item['weight']
            items[id, 1] = item['value']

        return backpacks, items


    def validate_history(self):
        for problem in self.history:
            for backpack in problem:
                if backpack.is_valid() == False:
                    return False

        return True

if __name__ == "__main__":
    env_name = 'Knapsack'

    with open(f"configs/KnapsackV2.json") as json_file:
        params = json.load(json_file)

    env_config = params['env_config']

    env = KnapsackV2(env_name, env_config)

    env.generate_dataset()
    env.generate_batch()
    env.generate_masks()

    # backpack_ids = [0 , 1]
    # item_ids = [3, 4]

    # next_step, rewards, isDone, info = env.step(backpack_ids, item_ids)
    # env.print_history()

    # state, backpack_net_mask, item_net_mask = env.reset()

    # items = state[[0,1], [3,4]]
    # items[0][0] = 999
    
    state = np.array(
        [[[-2, -2],
          [14,  0],
          [39,  1],
          [19, 60],
          [ 1, 13]]],
          dtype='float32' 
    )
    
    items = np.array(
        [[19, 60]],
        dtype='float32'
        )
    items_mask = np.array(
        [[1., 1., 1., 0., 1.]],
        dtype='float32'
        )
    bp_mask = np.array(
        [[0., 0., 0., 1., 1.]],
        dtype='float32'
    )

    new_mask = env.build_feasible_mask(state, items, bp_mask)
    print(new_mask)

    env.convert_to_ortools_input()
