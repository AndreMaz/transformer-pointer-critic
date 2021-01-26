
import numpy as np

class Resource():
    def __init__(self,
                batch_id,
                id,
                resource_representation):
        super(Resource, self).__init__()

        self.batch_id = batch_id
        self.id = id
        self.CPU = np.array([resource_representation[0]], dtype='float32')
        self.RAM = np.array([resource_representation[1]], dtype='float32')
        self.MEM = np.array([resource_representation[2]], dtype='float32')

    def get_tensor_rep(self):

        return np.asanyarray([
            self.CPU,
            self.RAM,
            self.MEM
        ]).flatten()        

    def print(self):
        print(f'Batch: {self.batch_id} \t| ID : {self.id:02d} \t| CPU: {self.CPU[0]:1f} \t| RAM: {self.RAM[0]:1f} \t| MEM: {self.MEM[0]:1f} \t')