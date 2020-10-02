
class Resource():
    def __init__(self,
                id,
                CPU,
                RAM,
                MEM,
                task,
                request_type):
        super(Resource, self).__init__()

        self.id = id
        self.CPU = CPU
        self.RAM = RAM
        self.MEM = MEM
        self.task = task
        self.request_type = request_type
        
