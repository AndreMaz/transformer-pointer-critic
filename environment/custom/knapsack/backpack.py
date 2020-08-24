
class Backpack():
    def __init__(self, id, capacity):
        super(Backpack, self).__init__()

        self.id = id
        self.capacity = capacity
        self.current_load = 0
        self.current_value = 0

        self.items = []
    
    def add_item(self, id, weight, value):
        item = {
            'id': id,
            'weight': weight,
            'value': value
        }
        
        if (self.id != 0):
            assert self.capacity >= self.current_load + weight,\
                f'Backpack {self.id} is overloaded. Maximum capacity: {self.capacity} || Item Weight: {weight}'

            self.current_load = self.current_load + weight
            self.current_value = self.current_value + value

        self.items.append(item)
    
    def reset(self):
        self.current_load = 0
        self.items = []
    
    def print(self):
        print(f'Backpack ID: {self.id} | Maximum Capacity: {self.capacity} | Current Load: {self.current_load} | Backpack Value: {self.current_value}')

        print(f'Items in the backpack:')
        if len(self.items) == 0: print('<Empty>')
        for item in self.items:
            print(f'ID: {item["id"]} | Weight {item["weight"]} | Value {item["value"]}')
    
    def is_valid(self):
        total_load = 0
        for i in self.items:
            total_load += i['weight']

        # By default EOS is always true
        if (self.id == 0): return True
        
        assert total_load == self.current_load, 'Total weight of items is different from current load of backpack'

        if total_load > self.capacity:
            return False
        
        return True