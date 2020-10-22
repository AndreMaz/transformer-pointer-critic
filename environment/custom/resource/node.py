import sys

sys.path.append('.')

from environment.custom.resource.resource import Resource
from environment.custom.resource.penalty import Penalty
import numpy as np

class Node():
    def __init__(self,
                 batch_id,
                 id,
                 CPU,
                 RAM,
                 MEM,
                 lower_task,
                 upper_task,
                 penalizer: Penalty):
        super(Node, self).__init__()

        self.batch_id = batch_id
        self.id = id
        self.CPU = np.array([CPU], dtype='float32')
        self.RAM = np.array([RAM], dtype='float32')
        self.MEM = np.array([MEM], dtype='float32')

        self.remaining_CPU = np.array([CPU], dtype='float32')
        self.remaining_RAM = np.array([RAM], dtype='float32')
        self.remaining_MEM = np.array([MEM], dtype='float32')

        self.penalizer = penalizer

        self.lower_task = np.array([lower_task], dtype='float32')
        self.upper_task = np.array([upper_task], dtype='float32')

        self.resources = []

    def add_resource(self,
                     id,
                     CPU,
                     RAM,
                     MEM,
                     task,
                     request_type):

        req = Resource(id, CPU, RAM, MEM, task, request_type)

        remaning_CPU = 0
        remaning_RAM = 0
        remaining_MEM = 0

        if (self.id != 0):
            isValid, remaning_CPU, remaning_RAM, remaining_MEM = self.validate(req)

            assert isValid == True,\
                f'Node {self.id} is overloaded. Cannot Place Resource {CPU}|{RAM}|{MEM} to a Node with {self.remaining_CPU}/{self.CPU}|{self.remaining_RAM}/{self.RAM}|{self.remaining_MEM}/{self.MEM}'

            self.remaining_CPU = remaning_CPU
            self.remaining_RAM = remaning_RAM
            self.remaining_MEM = remaining_MEM

        self.resources.append(req)

        return [remaning_CPU, remaning_RAM, remaining_MEM]

    def reset(self):
        self.remaining_CPU = self.CPU
        self.remaining_RAM = self.RAM
        self.remaining_MEM = self.MEM

        self.resources = []

    def validate(self, resource: Resource):
        
        CPU = 0
        RAM = 0
        MEM = 0

        if self.penalizer.to_penalize(self.lower_task, self.upper_task, resource.task) == False:
            CPU = self.remaining_CPU - resource.CPU
            RAM = self.remaining_RAM - resource.RAM
            MEM = self.remaining_MEM - resource.MEM
        else:
            CPU = self.remaining_CPU - self.penalizer.compute_CPU_penalty(resource.CPU)
            RAM = self.remaining_RAM - self.penalizer.compute_RAM_penalty(resource.RAM)
            MEM = self.remaining_MEM - self.penalizer.compute_MEM_penalty(resource.MEM)

        if CPU >= 0 and RAM >= 0 and MEM >= 0:
            return True, CPU, RAM, MEM
        else:
            return False, CPU, RAM, MEM

    def print(self):
        print(f'Node ID: {self.id} \t| Remaining CPU: {np.around(self.remaining_CPU, decimals=4)} of {self.CPU} \t| Remaining RAM: {np.around(self.remaining_RAM, decimals=4)} of {self.RAM} \t| Remaining MEM: {np.around(self.remaining_MEM, decimals=4)} of {self.MEM} \t| Lower Task: {int(self.lower_task[0])} \t| Upper Task: {int(self.upper_task[0])}')
        
        print('Resources allocated to the Node:')
        if len(self.resources) == 0: print('<Empty>')
        for res in self.resources:
            res.print()

    def is_valid(self):
        if self.remaining_CPU >= 0 and self.remaining_RAM >= 0 and self.remaining_MEM >= 0:
            return True
        
        return False

if __name__ == "__main__":
    batch_id = 0
    id = 2
    CPU = 10
    RAM = 10
    MEM = 10
    lower_task = 1
    upper_task = 3

    CPU_misplace_penalty = 10 
    RAM_misplace_penalty = 10
    MEM_misplace_penalty = 10

    penalty = Penalty(
        CPU_misplace_penalty,
        RAM_misplace_penalty,
        MEM_misplace_penalty
    )

    node = Node(
        batch_id,
        id,
        CPU,
        RAM,
        MEM,
        lower_task,
        upper_task,
        penalty,
    )

    resource_id = 1
    resource_CPU = 10
    resource_RAM  = 100
    resource_MEM = 10
    resource_task = 123
    request_type = 1


    node.add_resource(
        resource_id,
        resource_CPU,
        resource_RAM,
        resource_MEM,
        resource_task,
        request_type
    )
