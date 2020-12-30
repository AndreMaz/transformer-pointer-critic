from os import stat
import sys
from typing import List

sys.path.append('.')

from environment.custom.resource_v2.resource import Resource as Request
import numpy as np

class Node():
    def __init__(self,
                 batch_id,
                 id,
                 CPU,
                 RAM,
                 MEM,
                 ):
        super(Node, self).__init__()

        self.batch_id = batch_id
        self.id = id
        self.CPU = np.array([CPU], dtype='float32')
        self.RAM = np.array([RAM], dtype='float32')
        self.MEM = np.array([MEM], dtype='float32')

        self.remaining_CPU = self.CPU
        self.remaining_RAM = self.RAM
        self.remaining_MEM = self.MEM

        self.req_list: List[Request] = []

        # History stats
        self.CPU_history = [self.CPU]
        self.RAM_history = [self.RAM]
        self.MEM_history = [self.MEM]
        
    def reset(self):
        self.remaining_CPU = self.CPU
        self.remaining_RAM = self.RAM
        self.remaining_MEM = self.MEM

        self.req_list: List[Request] = []

        self.CPU_history = [self.CPU]
        self.RAM_history = [self.RAM]
        self.MEM_history = [self.MEM]

    def insert_req(self, req: Request):

        # Update the remaining resources of the Node
        self.remaining_CPU -= req.CPU
        self.remaining_RAM -= req.RAM
        self.remaining_MEM -= req.MEM

        self.req_list.append(req)

        self.CPU_history.append(self.remaining_CPU)
        self.RAM_history.append(self.remaining_RAM)
        self.MEM_history.append(self.remaining_MEM)
        
    
    def compute_node_load(self):
        CPU_load = (1 - self.remaining_CPU / self.CPU) * 100
        RAM_load = (1 - self.remaining_RAM / self.RAM) * 100
        MEM_load = (1 - self.remaining_MEM / self.MEM) * 100

        return CPU_load, RAM_load, MEM_load

if __name__ == "__main__":
    a = 1