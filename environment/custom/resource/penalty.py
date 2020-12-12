import tensorflow as tf
import numpy as np

def PenaltyFactory(opts: dict, EOS_CODE, resource_normalization_factor):
    penalties = {
        'greedy': GreedyPenalty,
    }

    try:
        penaltyType = opts['type']
        P = penalties[f'{penaltyType}']
        return P(opts[f'{penaltyType}'], EOS_CODE, resource_normalization_factor)
    except KeyError:
        raise NameError(f'Unknown Penalizer Name! Select one of {list(penalties.keys())}')

class GreedyPenalty():
    def __init__(self,
                 opts,
                 EOS_CODE,
                 resource_normalization_factor
                 ):
        super(GreedyPenalty, self).__init__()

        self.EOS_CODE = EOS_CODE

        self.CPU_penalty = np.array(
            [opts['CPU_misplace_penalty'] / resource_normalization_factor],
            dtype='float32'
        )
        self.RAM_penalty = np.array(
            [opts['RAM_misplace_penalty'] / resource_normalization_factor],
            dtype='float32'
        )
        self.MEM_penalty = np.array(
            [opts['MEM_misplace_penalty'] / resource_normalization_factor],
            dtype='float32'
        )

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
    
    def to_penalize_batch(self, bin_lower_type, bin_upper_type, resource_type, batch_size):
        
        # resource_type = np.reshape(resource_type, (batch_size, 1))

        # Look for places that won't be penalized
        # From: https://stackoverflow.com/a/55870586
        lower_tensor = tf.greater_equal(resource_type, bin_lower_type)
        upper_tensor = tf.less_equal(resource_type, bin_upper_type)

        in_range = tf.logical_and(lower_tensor, upper_tensor)
        # Marked as 1 = there's penalty involved
        # Marked as 0 = no penalty
        in_range = 1 - tf.cast(in_range, dtype='int32')

        # in_range = np.reshape(in_range, (batch_size, 1))

        return in_range

    def tensor_representation(self):
        return self.tensor
