class History():
    def __init__(self, id, capacity):
        super(History, self).__init__()

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
                f'Backpack {id} is overloaded. Maximum capacity: {self.capacity} || Item Weight: {weight}'

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