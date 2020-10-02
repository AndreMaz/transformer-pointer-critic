import sys

sys.path.append('.')

from environment.custom.resource.resource import Resource
from environment.custom.resource.penalty import Penalty

class Node():
    def __init__(self,
                 batch_id,
                 id,
                 CPU,
                 RAM,
                 MEM,
                 lower_task,
                 upper_task,
                 penalty: Penalty):
        super(Node, self).__init__()

        self.batch_id = batch_id
        self.id = id
        self.CPU = CPU
        self.RAM = RAM
        self.MEM = MEM

        self.used_CPU = 0
        self.used_RAM = 0
        self.used_MEM = 0

        self.penalty = penalty


        self.lower_task = lower_task
        self.upper_task = upper_task

        self.resources = []

    def add_resource(self,
                     id,
                     CPU,
                     RAM,
                     MEM,
                     task,
                     request_type):

        req = Resource(id, CPU, RAM, MEM, task, request_type)

        return self.is_valid(req)

        if (self.id != 0):
            assert self.is_valid(req, misplace_penalty) == True,\
                f'Node {self.id} is overloaded. Maximum capacity: {self.capacity} || Item Weight: {weight}'

        self.resources.append(req)

    def reset(self):
        self.used_CPU = 0
        self.used_RAM = 0
        self.used_MEM = 0

        self.resources = []

    def is_valid(self, resource: Resource):

        isValid = False

        if self.lower_task <= resource.task <= self.upper_task:
            if self.used_CPU + resource.CPU <= self.CPU\
                    and self.used_RAM + resource.RAM <= self.RAM\
                    and self.used_MEM + resource.MEM <= self.MEM:
                return True
        else:
            if self.used_CPU + self.penalty.compute_CPU_penalty(resource.CPU) <= self.CPU\
                    and self.used_RAM + self.penalty.compute_RAM_penalty(resource.RAM) <= self.RAM\
                    and self.used_MEM + self.penalty.compute_MEM_penalty(resource.MEM) <= self.MEM:
                return True

        return isValid


if __name__ == "__main__":
    batch_id = 0
    id = 0
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
    resource_RAM  = 10
    resource_MEM = 10
    resource_task = 123
    request_type = 1


    print(node.add_resource(
        resource_id,
        resource_CPU,
        resource_RAM,
        resource_MEM,
        resource_task,
        request_type
    ))
