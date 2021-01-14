from os import stat
import sys
from typing import List

sys.path.append('.')

from environment.custom.resource_v3.resource import Resource as Request
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

        self.remaining_CPU = self.CPU.copy()
        self.remaining_RAM = self.RAM.copy()
        self.remaining_MEM = self.MEM.copy()

        self.req_list: List[Request] = []

        # History stats
        self.CPU_history = [self.CPU]
        self.RAM_history = [self.RAM]
        self.MEM_history = [self.MEM]
        
    def reset(self):
        self.remaining_CPU = self.CPU.copy()
        self.remaining_RAM = self.RAM.copy()
        self.remaining_MEM = self.MEM.copy()

        self.req_list: List[Request] = []

        self.CPU_history = [self.CPU]
        self.RAM_history = [self.RAM]
        self.MEM_history = [self.MEM]

    def insert_req(self, req: Request):

        if self.id != 0:
            # Update the remaining resources of the Node
            self.remaining_CPU -= req.CPU
            self.remaining_RAM -= req.RAM
            self.remaining_MEM -= req.MEM

            self.CPU_history.append(self.remaining_CPU.copy())
            self.RAM_history.append(self.remaining_RAM.copy())
            self.MEM_history.append(self.remaining_MEM.copy())

        self.req_list.append(req)

    def print(self, print_details = False):
        CPU_load = np.around(self.remaining_CPU, decimals=4)
        RAM_load = np.around(self.remaining_RAM, decimals=4)
        MEM_load = np.around(self.remaining_MEM, decimals=4)
        dominant = min(self.remaining_CPU, self.remaining_RAM, self.remaining_MEM)

        print(f'Node ID: {self.id} \t| Remaining CPU: {CPU_load} of {self.CPU} \t| Remaining RAM: {RAM_load} of {self.RAM} \t| Remaining MEM: {MEM_load} of {self.MEM} || Dominant {dominant[0]:.3f} \t')
        
        total_nodes = len(self.req_list)

        if print_details:
            print('Resources allocated to the Node:')
            if total_nodes == 0: print('<Empty>')
            for res in self.req_list:
                res.print()
        
        self.print_history_stats()

        print(f'Total Requests {total_nodes}.')

    def print_history_stats(self):
        print('CPU History')

        np.set_printoptions(precision=4)

        print(np.asanyarray(self.CPU_history).flatten())
        
        print('RAM History')
        print(np.asanyarray(self.RAM_history).flatten())

        print('MEM History')
        print(np.asanyarray(self.MEM_history).flatten())    

    def get_tensor_rep(self):

        return np.asanyarray([
            self.remaining_CPU,
            self.remaining_RAM,
            self.remaining_MEM
        ]).flatten()

if __name__ == "__main__":
    a = 1