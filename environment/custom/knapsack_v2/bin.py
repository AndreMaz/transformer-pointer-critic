import sys
from typing import List

sys.path.append('.')

from environment.custom.knapsack_v2.item import Item
from environment.custom.knapsack_v2.misc.utils import round_half_up
import numpy as np

class Bin():
    def __init__(self,
                 batch_id,
                 id,
                 bin_representation
                 ):
        super(Bin, self).__init__()

        self.batch_id = batch_id
        self.id = id
        self.capacity = np.array([bin_representation[0]], dtype='float32')
        self.current_load = np.array([bin_representation[1]], dtype='float32')

        self.item_list: List[Item] = []
        
        self.current_value = np.array([0.0], dtype="float32")

        # History stats
        self.load_history = [self.current_load]
    
    def reset(self):
        self.current_load = np.array([0], dtype="float32")

        self.item_list = []

        self.load_history = [self.current_load]

        self.current_value = np.array([0.0], dtype="float32")

    def compute_updated_load(self, item: Item):
        return round_half_up(self.current_load + item.weight, 2)

    def can_fit_item(self, item: Item):
        updated_load = self.compute_updated_load(item)

        return round_half_up(self.capacity - updated_load, 2) >= 0
    
    def insert_item(self, item: Item):
        
        if self.id != 0:
            assert self.can_fit_item(item),\
                f'Bin {self.id} is overloaded. Current load {self.current_load[0]:.2f}/{self.capacity[0]:.2f} || Item: weight {item.weight[0]:.2f} value {item.value[0]:.2f}'

            self.current_load = self.compute_updated_load(item)

            self.load_history.append(self.current_load.copy())

        self.current_value += item.value
        self.item_list.append(item)
    
    def print(self, print_details = False): # pragma: no cover
        maximum_capacity = np.around(self.capacity, decimals=4)
        current_load = np.around(self.current_load, decimals=4)

        print(f'Node ID: {self.id} \t| Current Load {current_load} of {maximum_capacity}')

        total_items = len(self.item_list)

        if print_details:
            print('Resources allocated to the Node:')
            if total_items == 0: print('<Empty>')
            for res in self.item_list:
                res.print()
        
        print(f'Total Requests {total_items}.')

    def get_tensor_rep(self):

        return np.asanyarray([
            self.capacity,
            self.current_load,
        ]).flatten()