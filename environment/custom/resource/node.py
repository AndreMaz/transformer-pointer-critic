from os import stat
import sys
from typing import List

sys.path.append('.')

from environment.custom.resource.resource import Resource
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
                 penalizer,
                 task_normalization_factor,
                 gather_stats = False
                 ):
        super(Node, self).__init__()

        self.gather_stats: bool = gather_stats

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

        self.resources: List[Resource] = []

        self.task_normalization_factor = task_normalization_factor
        
        # History stats
        self.CPU_history = [self.CPU]
        self.RAM_history = [self.RAM]
        self.MEM_history = [self.MEM]
        self.percentage_penalized_history = [0.0]

        self.premium_reqs = 0
        self.free_reqs = 0

    def add_resource(self,
                     iteration_num,
                     id,
                     CPU,
                     RAM,
                     MEM,
                     task,
                     request_type):

        req = Resource(iteration_num, id, CPU, RAM, MEM, task, request_type)

        return self.insert_resource(req)

    def insert_resource(self, req: Resource):

        remaning_CPU = 0
        remaning_RAM = 0
        remaining_MEM = 0

        if (self.id != 0):
            isValid, remaning_CPU, remaning_RAM, remaining_MEM = self.validate(req)

            assert isValid == True,\
                f'Node {self.id} is overloaded. Cannot Place Resource {req.CPU}|{req.RAM}|{req.MEM} to a Node with {self.remaining_CPU}/{self.CPU}|{self.remaining_RAM}/{self.RAM}|{self.remaining_MEM}/{self.MEM}'

            self.remaining_CPU = remaning_CPU
            self.remaining_RAM = remaning_RAM
            self.remaining_MEM = remaining_MEM

        self.resources.append(req)

        # History stats
        if self.gather_stats:
            self.CPU_history.append(self.remaining_CPU)
            self.RAM_history.append(self.remaining_RAM)
            self.MEM_history.append(self.remaining_MEM)
            self.percentage_penalized_history.append(self.compute_percentage_penalized_resources())

        if req.request_type == 1:
            self.premium_reqs += 1
        else:
            self.free_reqs += 1

        return [remaning_CPU, remaning_RAM, remaining_MEM]


    def reset(self):
        self.remaining_CPU = self.CPU
        self.remaining_RAM = self.RAM
        self.remaining_MEM = self.MEM

        self.resources = []

        self.CPU_history = [self.CPU]
        self.RAM_history = [self.RAM]
        self.MEM_history = [self.MEM]
        self.percentage_penalized_history = [0.0]

        self.premium_reqs = 0
        self.free_reqs = 0

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

    def print(self, print_details = False):
        if self.id == 0:
            print(f'Node ID: {self.id} \t| Remaining CPU: {np.around(self.remaining_CPU, decimals=4)} of {self.CPU} \t| Remaining RAM: {np.around(self.remaining_RAM, decimals=4)} of {self.RAM} \t| Remaining MEM: {np.around(self.remaining_MEM, decimals=4)} of {self.MEM} \t| Lower Task: {int(self.lower_task[0])} \t| Upper Task: {int(self.upper_task[0])}')
        else:
            print(f'Node ID: {self.id} \t| Remaining CPU: {np.around(self.remaining_CPU, decimals=4)} of {self.CPU} \t| Remaining RAM: {np.around(self.remaining_RAM, decimals=4)} of {self.RAM} \t| Remaining MEM: {np.around(self.remaining_MEM, decimals=4)} of {self.MEM} \t| Lower Task: {self.lower_task[0]:1f} \t| Upper Task: {self.upper_task[0]:1f}')
        
        if print_details:
            print('Resources allocated to the Node:')
            if len(self.resources) == 0: print('<Empty>')
            for res in self.resources:
                res.print()

        total_nodes = len(self.resources)
        if total_nodes == 0 or self.id == 0: 
            print(f'Total Requests {total_nodes}. Premium {self.premium_reqs} and Free {self.free_reqs}. Percentage of penalized resources: {0:.2f}%')
            return

        # within_range = 0

        # low = int(round(self.lower_task[0] * self.task_normalization_factor))
        # up = int(round(self.upper_task[0] * self.task_normalization_factor))

        # for res in self.resources:
        #     task = int(round(res.task[0] * self.task_normalization_factor))
        #     if low <= task <= up:
        #         within_range += 1

        percentage_penalized = self.compute_percentage_penalized_resources()

        print(f'Total Requests {total_nodes}. Premium {self.premium_reqs} and Free {self.free_reqs}. Percentage of penalized resources: {percentage_penalized:.2f}%')
        
        CPU_load, RAM_load, MEM_load = self.compute_node_load()
        print(f'Load CPU {CPU_load[0]:.2f}% | RAM {RAM_load[0]:.2f}% | MEM {MEM_load[0]:.2f}%')
    
    def compute_node_load(self):
        CPU_load = (1 - self.remaining_CPU / self.CPU) * 100
        RAM_load = (1 - self.remaining_RAM / self.RAM) * 100
        MEM_load = (1 - self.remaining_MEM / self.MEM) * 100

        return CPU_load, RAM_load, MEM_load

    def compute_percentage_penalized_resources(self):
        total_nodes = len(self.resources)

        if total_nodes == 0 or self.id == 0: return 0.0

        within_range = 0

        low = int(round(self.lower_task[0] * self.task_normalization_factor))
        up = int(round(self.upper_task[0] * self.task_normalization_factor))

        for res in self.resources:
            task = int(round(res.task[0] * self.task_normalization_factor))
            if low <= task <= up:
                within_range += 1

        percentage_penalized = (1 - (within_range/total_nodes))*100

        return percentage_penalized

    def print_history_stats(self):
        print('CPU History')
        print(np.asanyarray(self.CPU_history).flatten())
        
        print('RAM History')
        print(np.asanyarray(self.RAM_history).flatten())

        print('MEM History')
        print(np.asanyarray(self.MEM_history).flatten())
    
    def is_valid(self):
        if self.remaining_CPU >= 0 and self.remaining_RAM >= 0 and self.remaining_MEM >= 0:
            return True
        
        return False

    def get_stats(self) -> dict:
        stats = {}

        stats['node_id'] = self.id
        stats['num_resources'] = len(self.resources)
        stats['percentage_penalized_resources'] = self.compute_percentage_penalized_resources()

        stats['CPU_History'] = np.asanyarray(self.CPU_history).flatten()
        stats['RAM_History'] = np.asanyarray(self.RAM_history).flatten()
        stats['MEM_History'] = np.asanyarray(self.MEM_history).flatten()
        
        CPU_load, RAM_load, MEM_load = self.compute_node_load()

        stats['CPU_load'] = CPU_load
        stats['RAM_load'] = RAM_load
        stats['MEM_load'] = MEM_load

        stats['premium_reqs'] = self.premium_reqs
        stats['free_reqs'] = self.free_reqs

        return stats
    
    def get_rejection_stats(self) -> dict:
        
        stats = {}

        total_nodes = len(self.resources)
        num_premium_rejected = 0
        batch_premium_rejected = 0

        num_free_rejected = 0
        batch_free_rejected = 0

        stats['total_nodes'] = total_nodes

        if total_nodes != 0:
            for r in self.resources:
                if r.request_type == 0:
                    num_free_rejected += 1

                    # Only update once
                    if batch_free_rejected == 0:
                        batch_free_rejected = r.batch_id

                if r.request_type == 1:
                    num_premium_rejected += 1
                    # Only update once
                    if batch_premium_rejected == 0:
                        batch_premium_rejected = r.batch_id


        stats['num_premium_rejected'] = num_premium_rejected
        stats['batch_premium_rejected'] = batch_premium_rejected
        stats['num_free_rejected'] = num_free_rejected
        stats['batch_free_rejected'] = batch_free_rejected

        return stats

if  __name__ == "__main__": # pragma: no cover
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
