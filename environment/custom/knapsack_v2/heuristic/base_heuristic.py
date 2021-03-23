
from typing import List, Tuple
from environment.custom.knapsack_v2.bin import Bin
from environment.custom.knapsack_v2.item import Item

class BaseHeuristic():
    def __init__(self,
                num_bins:int
                ):
        self.name = 'base_heuristic'

        self.resource_batch_id = 0
        self.num_bins = num_bins

        self.solution = []

    def generate_name(self, state):
        raise NotImplementedError('Method "generate_name" not implemented')

    def solve(self, state):
        raise NotImplementedError('Method "solve" not implemented')

    def place_single_item(self, resource, node_list, EOS_NODE):
        raise NotImplementedError('Method "place_single_resource" not implemented')

    def reset(self):
        self.resource_batch_id = 0
        
        self.solution = []

    def parse_bins(self, state) -> List[Bin]:

        batch_size = state.shape[0]

        assert batch_size == 1, 'Heuristic only works for problems with batch size equal to 1!'

        bins = state[0, :self.num_bins, :]
        # resources = state[:, env.bin_sample_size:, :]
        
        bin_list = []
        for id, bin in enumerate(bins):
            bin_list.append(
                Bin(
                    0, # Nodes are created in first batch, i.e., state from env
                    id,
                    bin
                )
            )
        
        return bin_list

    def parse_items(self, state) -> List[Item]:
        batch_size = state.shape[0]

        assert batch_size == 1, 'Heuristic only works for problems with batch size equal to 1!'

        items = state[0, self.num_bins:, :]

        item_list = []
        for id, item in enumerate(items):
            item_list.append(
                Item(
                    self.resource_batch_id,
                    id,
                    item
                )
            )

        self.resource_batch_id += 1
        
        return item_list
    

    def print_info(self, elem_list: list): # pragma: no cover
        for elem in elem_list:
            elem.print()
            # print(elem.get_stats())

    def print_node_stats(self, print_details = False): # pragma: no cover
        for node in self.solution:
            node.print(print_details)
