from random import randint

class Item():
    def __init__(self,
                 id: int,
                 value,
                 weight,
                 ):

        self.id = id

        self.value = value
        self.weight = weight

        self.ratio = value / weight

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