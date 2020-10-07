
class Penalty():
    def __init__(self,
                 CPU_misplace_penalty,
                 RAM_misplace_penalty,
                 MEM_misplace_penalty):
        super(Penalty, self).__init__()

        self.CPU_penalty = CPU_misplace_penalty
        self.RAM_penalty = RAM_misplace_penalty
        self.MEM_penalty = MEM_misplace_penalty

    def compute_penalty(self, CPU, RAM, MEM):
        
        CPU = self.compute_CPU_penalty(CPU)
        RAM = self.compute_RAM_penalty(RAM)
        MEM = self.compute_MEM_penalty(MEM)

        return CPU, RAM, MEM

    def compute_CPU_penalty(self, CPU):
        return CPU + self.CPU_penalty

    def compute_RAM_penalty(self, RAM):
        return RAM + self.RAM_penalty

    def compute_MEM_penalty(self, MEM):
        return MEM + self.MEM_penalty
    
    def to_penalize(self, bin_lower_type, bin_upper_type, resource_type):

        if bin_lower_type <= resource_type <= bin_upper_type:
            return False
        
        return True