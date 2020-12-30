
import numpy as np

class Resource():
    def __init__(self,
                batch_id,
                id,
                CPU,
                RAM,
                MEM,):
        super(Resource, self).__init__()

        self.batch_id = batch_id
        self.id = id
        self.CPU = np.array([CPU], dtype='float32')
        self.RAM = np.array([RAM], dtype='float32')
        self.MEM = np.array([MEM], dtype='float32')
        

    def print(self):
        print(f'Batch: {self.batch_id} \t| ID : {self.id:02d} \t| CPU: {self.CPU[0]:1f} \t| RAM: {self.RAM[0]:1f} \t| MEM: {self.MEM[0]:1f} \t| Task: {self.task[0]:1f} \t| User Type: {int(self.request_type[0])}')