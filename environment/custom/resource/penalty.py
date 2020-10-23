import tensorflow as tf
import numpy as np

class Penalty():
    def __init__(self,
                 CPU_misplace_penalty,
                 RAM_misplace_penalty,
                 MEM_misplace_penalty,
                 EOS_CODE
                 ):
        super(Penalty, self).__init__()

        self.EOS_CODE = EOS_CODE

        self.CPU_penalty = np.array([CPU_misplace_penalty], dtype='float32')
        self.RAM_penalty = np.array([RAM_misplace_penalty], dtype='float32')
        self.MEM_penalty = np.array([MEM_misplace_penalty], dtype='float32')

        self.tensor = tf.constant([
            self.CPU_penalty,
            self.RAM_penalty,
            self.MEM_penalty
        ],
            dtype='float32',
            shape=(1, 3)
        )

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
        
        # EOS Node is always available
        if bin_lower_type == self.EOS_CODE and bin_upper_type == self.EOS_CODE:
            return False

        if bin_lower_type <= resource_type <= bin_upper_type:
            return False
        
        return True
    
    def tensor_representation(self):
        return self.tensor
