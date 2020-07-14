from random import randint
from environment.custom.knapsack.item import Item

EOS_BACKPACK = 'eos'
NORMAL_BACKPACK = 'backpack'

class Backpack():
    def __init__(self,
                 id: int,
                 normalization_factor: int,
                 backpack_type: str, # can be "normal" or "eos"
                 capacity: int = None,
                 min_capacity: int = None,
                 max_capacity: int = None):

        self.id = id
        self.normalization_factor = normalization_factor
        self.type = backpack_type

        if capacity is not None:
            self.capacity = capacity
        else:
            self.capacity = randint(min_capacity, max_capacity) / self.normalization_factor

        self.current_capacity = 0
        self.current_value = 0

        self.items = []

    def add_item(self, item: Item):
        """Places an item in backpack

        Args:
            item (Item): Item

        Returns:
            [tuple(boolean, int)]: If (True/Value) then value is current capacity. If (False/Value) then value represents backpack's overload.
        """
        # Mark item as taken
        item.take()
        
        # Only the normal backpack has weight
        # Update it
        if self.type == NORMAL_BACKPACK:
            self.current_capacity += item.weight
            self.current_value += item.value

        # Append the item    
        self.items.append(item)
        
        return self.is_valid()
    
    def clear(self):
        self.current_capacity = 0
        self.current_value = 0

        for item in self.items:
            # Mark item as not taken
            item.place_back()
        
        self.items = []

    def is_full(self):
        # Empty back pack is never full
        if self.type == EOS_BACKPACK: return False

        return self.current_capacity >= self.capacity

    def is_valid(self):
        is_valid = True
        capacity_diff = abs(self.current_capacity - self.capacity)

        if (self.current_capacity > self.capacity):
            is_valid = False

        return is_valid, capacity_diff, self.current_value

    def print_stats(self):
        print(f'Backpack ID: {self.id} | Type: {self.type} | Maximum Capacity: {self.capacity} | Current Load: {self.current_capacity} | Backpack Value: {self.current_value}')

        print(f'Items in the backpack:')
        if len(self.items) == 0: print('<Empty>')
        for item in self.items:
            item.print_stats()

