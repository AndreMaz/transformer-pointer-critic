
from environment.custom.resource.penalty import Penalty


class Reward():
    def __init__(self,
                 reward_per_level,
                 misplace_reward_penalty,
                 penalty
                 ):
        super(Reward, self).__init__()

        self.reward_per_level = reward_per_level
        self.misplace_reward_penalty = misplace_reward_penalty
        self.penalty: Penalty = penalty

    def compute_reward(self,
                       batch,
                       total_num_nodes,
                       bin,
                       resource,
                       ):

        bins = batch[:total_num_nodes]
        resources = batch[total_num_nodes:]

        bin_remaning_CPU = bin[0]
        bin_remaning_RAM = bin[1]
        bin_remaning_MEM = bin[2]
        bin_lower_type = bin[3]
        bin_upper_type = bin[4]

        resource_CPU = resource[0]
        resource_RAM = resource[1]
        resource_MEM = resource[2]
        resource_type = resource[3]
        request_type = int(resource[4])
        
        reward = 0
        if self.penalty.toPenalize(bin_lower_type,bin_upper_type, resource_type):
            reward = self.reward_per_level[request_type] - self.misplace_reward_penalty
        else:
            reward = self.reward_per_level[request_type]
            
        return reward
