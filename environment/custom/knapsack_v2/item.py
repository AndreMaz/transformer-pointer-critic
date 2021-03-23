
import numpy as np

class Item():
    def __init__(self,
                batch_id,
                id,
                item_representation):
        super(Item, self).__init__()

        self.batch_id = batch_id
        self.id = id
        self.weight = np.array([item_representation[0]], dtype='float32')
        self.value = np.array([item_representation[1]], dtype='float32')

        self.ratio = self.value / self.weight

    def get_tensor_rep(self):

        return np.asanyarray([
            self.weight,
            self.value,
        ]).flatten()        

    def print(self): # pragma: no cover
        print(f'Batch: {self.batch_id} \t| ID : {self.id:02d} \t| Weight: {self.weight[0]:1f} \t| RAM: {self.value[0]:1f} \t')