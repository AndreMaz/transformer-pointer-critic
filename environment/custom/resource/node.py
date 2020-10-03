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
                 penalizer: Penalty):
        super(Node, self).__init__()

        self.batch_id = batch_id
        self.id = id
        self.CPU = CPU
        self.RAM = RAM
        self.MEM = MEM

        self.remaining_CPU = CPU
        self.remaining_RAM = RAM
        self.remaining_MEM = MEM

        self.penalizer = penalizer


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

        # return self.is_valid(req)

        if (self.id != 0):
            assert self.is_valid(req) == True,\
                f'Node {self.id} is overloaded. Cannot Place Resource {CPU}|{RAM}|{MEM} to a Node with {self.used_CPU}/{self.CPU}|{self.used_RAM}/{self.RAM}|{self.used_MEM}/{self.MEM}'

            # if self.

        self.resources.append(req)

    def reset(self):
        self.remaining_CPU = self.CPU
        self.remaining_RAM = self.RAM
        self.remaining_MEM = self.MEM

        self.resources = []

    def is_valid(self, resource: Resource):

        isValid = False

        # if self.lower_task <= resource.task <= self.upper_task:
        if self.penalizer.toPenalize(self.lower_task, self.upper_task, resource.task) == False:
            if self.remaining_CPU - resource.CPU >= 0\
                    and self.remaining_RAM - resource.RAM >= 0\
                    and self.remaining_MEM - resource.MEM >= 0:
                return True
        else:
            if self.remaining_CPU - self.penalizer.compute_CPU_penalty(resource.CPU) >= 0\
                    and self.remaining_RAM - self.penalizer.compute_RAM_penalty(resource.RAM) >= 0\
                    and self.remaining_MEM + self.penalizer.compute_MEM_penalty(resource.MEM) >= 0:
                return True

        return isValid


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
