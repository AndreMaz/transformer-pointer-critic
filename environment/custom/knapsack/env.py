import json
import numpy as np

from environment.base.base import BaseEnvironment
from environment.custom.knapsack.item import Item
from environment.custom.knapsack.backpack import Backpack, EOS_BACKPACK, NORMAL_BACKPACK

class Knapsack(BaseEnvironment):
    def __init__(self, name: str, opts: dict):
        super(Knapsack, self).__init__(name)

        self.load_from_file = opts['load_from_file']
        self.location = opts['location']

        self.batch_dim = 1

        self.normalization_factor = opts['normalization_factor']

        self.new_items_on_reset = opts['new_items_on_reset']
        self.num_items = opts['num_items']
        self.num_backpacks = opts['num_backpacks']

        self.min_item_value = opts['min_item_value']
        self.max_item_value = opts['max_item_value']
        self.min_item_weight = opts['min_item_weight']
        self.max_item_weight = opts['max_item_weight']

        self.min_backpack_capacity = opts['min_backpack_capacity']
        self.max_backpack_capacity = opts['max_backpack_capacity']

        # Represents the number of backpacks (including EOS backpack) and items
        self.tensor_size = (self.num_backpacks + 1) + (self.num_items)

        if self.load_from_file:
            self.backpacks, self.items = self.load_problem()
        else:
            self.backpacks, self.items = self.generate()

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

    def reset(self):
        
        for backpack in self.backpacks.values():
            backpack.clear()

        if self.new_items_on_reset:
            _, self.items = self.generate()

        state = self.convert_to_tensor()

        backpack_net_mask, item_net_mask = self.compute_masks()

        return state, backpack_net_mask, item_net_mask

    def state(self):
        state = self.convert_to_tensor()

        backpack_net_mask, item_net_mask = self.compute_masks()

        return state, backpack_net_mask, item_net_mask

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
    
    def all_items_taken(self):
        all_taken = True

        for item in self.items.values():
            if not item.is_taken(): return False
        
        return all_taken

    def convert_to_tensor(self):
        """
        Converts the backpacks info and items info into a tensor of 
         [batch_size, num_backpacks + 1 (EOS backpack) + num_items, 2] shape.


        For example, 2 backpacks (with 10 and 5 capacities) and 2 items
            item 1: value 1, weight 2
            item 2: value 3, weight 3
        Will produce the following tensor of shape (1, 5, 2):
            index = 0: EOS backpack [ 0, 0]
            index = 1: 1 backpack   [10, 0]
            index = 2: 2 backpack   [ 5, 0]
            index = 3: 3 item       [ 2, 1]
            index = 4: 4 item       [ 3, 3]
        """

        tensor_env = np.zeros((self.batch_dim, self.tensor_size, 2), dtype='float32')

        for backpack in self.backpacks.values():
            if (backpack.type != 'eos'):
                tensor_env[0, backpack.id, 0] = backpack.capacity
                tensor_env[0, backpack.id, 1] = backpack.current_value

        for item in self.items.values():
            tensor_env[0, item.id, 0] = item.weight
            tensor_env[0, item.id, 1] = item.value
        
        return tensor_env

    def compute_masks(self):
        """Computes the masks for the item and backpack Pointer nets
        Returns:
            [List of Tensors]: Masks
        """

        # Represents positions marked as "0" where item Ptr Net can point
        item_net_mask = np.zeros((self.batch_dim, self.tensor_size), dtype='float16')
        # Represents positions marked as "0" where backpack Ptr Net can point
        backpack_net_mask = np.ones((self.batch_dim, self.tensor_size), dtype='float16')

        
        # Default mask
        for backpack in self.backpacks.values():
            item_net_mask[0, int(backpack.id)] = 1

        # Default mask
        backpack_net_mask = backpack_net_mask - item_net_mask

        # Iterate over the items that were already taken and mask them
        for item in self.items.values():
            if (item.is_taken()):
                item_net_mask[0, int(item.id)] = 1
        
        # # Iterate over the backpacks that are already full and mask them
        for backpack in self.backpacks.values():
            if (backpack.is_full()):
                backpack_net_mask[0, int(backpack.id)] = 1

        return backpack_net_mask, item_net_mask

    def add_stats_to_agent_config(self, agent_config):
        agent_config['num_items'] = self.num_items
        agent_config['num_backpacks'] = self.num_backpacks
        agent_config['tensor_size'] = self.tensor_size

        agent_config['vocab_size'] = self.tensor_size
    
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

    def convert_to_ortools_input(self):
        data = {}

        weights = []
        values = []
        for item in self.items.values():
            weights.append(item.weight)
            values.append(item.value)
        
        data['weights'] = weights
        data['values'] = values
        data['items'] = list(range(len(weights)))
        data['num_items'] = len(weights)

        backpacks = []
        for backpack in self.backpacks.values():
            backpacks.append(backpack.capacity)
        
        data['bin_capacities'] = backpacks
        
        data['bins'] = list(range(len(backpacks)))

        return data
    
    def is_valid(self):
        is_valid = True

        for backpack in self.backpacks.values():
            is_valid, _, _ = backpack.is_valid()
            if not is_valid: return False

        return is_valid

    def load_problem(self):
        with open(self.location) as json_file:
            problem = json.load(json_file)

        backpacks_dict: dict = problem['backpacks']
        backpacks = {}

        for backpack in backpacks_dict.values():
            backpacks[f'{backpack["id"]}'] = Backpack(
            backpack['id'],
            self.normalization_factor,
            backpack['type'],
            capacity = backpack['capacity']
        )


        items_dict = problem['items']
        items = {}

        for item in items_dict.values():
            items[f'{item["id"]}'] = Item(
            item['id'],
            self.normalization_factor,
            value = item['value'],
            weight = item['weight']
        )

        return backpacks, items

    def save_problem(self):
        backpacks = {}
        for backpack in self.backpacks.values():
            backpacks[f'{backpack.id}'] = {}
            backpacks[f'{backpack.id}']['id'] = backpack.id
            backpacks[f'{backpack.id}']['type'] = backpack.type
            backpacks[f'{backpack.id}']['capacity'] = backpack.capacity

        items = {}
        for item in self.items.values():
            items[f'{item.id}'] = {}
            items[f'{item.id}']['id'] = item.id
            items[f'{item.id}']['weight'] = item.weight
            items[f'{item.id}']['value'] = item.value

        problem = {
            "backpacks": backpacks,
            "items": items
        }

        with open(self.location, 'w') as fp:
            json.dump(problem, fp, indent=4)