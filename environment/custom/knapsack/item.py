from random import randint

class Item():
    def __init__(self,
                 id: int,
                 normalization_factor: int,
                 value: int = None,
                 weight: int = None,
                 min_value: int = None,
                 max_value: int = None,
                 min_weight: int = None,
                 max_weight: int = None
                 ):

        self.id = id

        self.normalization_factor = normalization_factor

        if value is not None: self.value = value
        else: self.value = randint(min_value, max_value) / self.normalization_factor

        if weight is not None: self.weight = weight
        else: self.weight = randint(min_weight, max_weight) / self.normalization_factor

        # By default item is not in a backpack, i.e., not taken
        self.taken = False

    def is_taken(self):
        return self.taken

    def take(self):
        if (self.taken): raise ValueError(f'Item {self.id} already taken')

        self.taken = True

    def place_back(self):
        if (self.taken == False): raise ValueError(f'Item {self.id} is not in the backpack. Cannot place it back')

        self.taken = False

    def print_stats(self):
        print(f'ID:{self.id} | Value:{self.value} | Weight:{self.weight} | Taken:{self.taken}')