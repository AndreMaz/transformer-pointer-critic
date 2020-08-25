from math import sqrt

class Vehicle():
    def __init__(self, id, depot_x, depot_y, capacity):
        super(Vehicle, self).__init__()

        self.id = id
        self.capacity = capacity
        self.depot_x = depot_x
        self.depot_y = depot_y

        self.current_x = depot_x
        self.current_y = depot_y

        self.current_load = 0
        self.route_distance = 0
        
        self.nodes = []

    def add_node(self, node_id, x, y, demand):
        node = {
            "id": node_id,
            "x": x,
            "y": y,
            "demand": demand
        }

        assert self.capacity >= self.current_load + demand, \
            f'Vehicle {self.id} is overloaded. Maximum capacity: {self.capacity} || Current Load {self.current_load} ||  Item Weight: {demand}'
        
        self.current_load += demand

        # Compute route distance so far
        xd = self.current_x - x
        yd = self.current_y - y

        distance = round(sqrt(xd*xd + yd*yd))

        self.route_distance += distance

        self.nodes.append(node)

    def reset(self):
        self.current_load = 0
        self.route_distance = 0
        self.nodes = []
        self.current_x = self.depot_x
        self.current_y = self.current_y
    
    def print(self):
        print(f'Vehicle ID: {self.id} | Maximum Capacity: {self.capacity} | Current Load: {self.current_load} | Route Distance: {self.route_distance}')
