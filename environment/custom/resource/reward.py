import numpy as np
import tensorflow_probability as tfp
import math

def RewardFactory(opts: dict, penalizer, EOS_CODE):
    rewards = {
        'greedy': GreedyReward,
        "fair": FairReward
    }

    try:
        rewardType = opts['type']
        R = rewards[f'{rewardType}']
        return R(opts[f'{rewardType}'], penalizer, EOS_CODE)
    except KeyError:
        raise NameError(f'Unknown Reward Name! Select one of {list(rewards.keys())}')


class GreedyReward():
    def __init__(self,
                 opts: dict,
                 penalizer,
                 EOS_CODE
                 ):
        super(GreedyReward, self).__init__()

        self.reward_per_level = opts['reward_per_level']
        self.misplace_penalty_factor = opts['misplace_penalty_factor']
        self.correct_place_factor = opts['correct_place_factor']
        self.penalizer = penalizer
        self.EOS_CODE = EOS_CODE

    def compute_reward(self,
                       batch,
                       total_num_nodes,
                       bin,
                       resource,
                       feasible_mask
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
        if self.penalizer.to_penalize(bin_lower_type, bin_upper_type, resource_type):
            reward = self.misplace_penalty_factor * self.reward_per_level[request_type]
        else:
            reward = self.correct_place_factor * self.reward_per_level[request_type]
        
        # If placed premium request at EOS while there were available nodes
        # Give negative reward
        if request_type == 1 and bin_upper_type == self.EOS_CODE:
            if not np.all(feasible_mask[1:] == 1):
                reward = -1 * reward

        # If free request is placed at EOS
        if request_type == 0 and bin_upper_type == self.EOS_CODE:
            reward = 0

        return reward


class FairReward():
    def __init__(self,
                 opts: dict,
                 penalizer,
                 EOS_CODE
                 ):
        super(FairReward, self).__init__()

        self.reward_per_level = opts['reward_per_level']
        self.misplace_penalty_factor = opts['misplace_penalty_factor']
        self.correct_place_factor = opts['correct_place_factor']
        self.penalizer = penalizer
        self.EOS_CODE = EOS_CODE

    def compute_reward(self,
                       batch,
                       total_num_nodes,
                       bin,
                       resource,
                       feasible_mask
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
        
        skewness_reward = 0
        if bin_lower_type != self.EOS_CODE and bin_upper_type != self.EOS_CODE:

            bin_resource_variance = tfp.stats.variance(bin[:3])
        
            if bin_resource_variance == 0:
                bin_resource_variance = 1
        
            skewness_reward = math.log(bin_resource_variance, 0.5) / 10

        reward = 0
        if self.penalizer.to_penalize(bin_lower_type, bin_upper_type, resource_type):
            reward = self.misplace_penalty_factor * (self.reward_per_level[request_type] + skewness_reward)
        else:
            reward = self.correct_place_factor * ( self.reward_per_level[request_type] + skewness_reward)
        
        # If placed premium request at EOS while there were available nodes
        # Give negative reward
        if request_type == 1 and bin_upper_type == self.EOS_CODE:
            if not np.all(feasible_mask[1:] == 1):
                reward = -1 * reward

        # If free request is placed at EOS
        if request_type == 0 and bin_upper_type == self.EOS_CODE:
            reward = 0

        return reward