import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import math
from environment.custom.resource.utils import bins_eos_checker, is_premium_wrongly_rejected_checker, bins_full_checker

def RewardFactory(opts: dict, penalizer, EOS_CODE):
    rewards = {
        'greedy': GreedyReward,
        "fair": FairReward
    }

    try:
        rewardType = opts['type']
        R = rewards[f'{rewardType}']
        print(f'"{rewardType.upper()}" reward selected.')
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

        self.premium_rejected = opts['premium_rejected']
        self.free_rejected = opts['free_rejected']

        self.mis_placement_tensor = tf.constant([
            self.correct_place_factor,
            self.misplace_penalty_factor,
        ],
            dtype='float32',
            shape=(1, 2)
        )

        self.reward_per_level_tensor = tf.constant(
            self.reward_per_level,
            dtype='float32',
            shape=(1,2)
        )

        self.rejection_tensor = tf.constant([
            self.premium_rejected,
            self.free_rejected
        ],
            dtype='float32',
            shape=(1, 2)
        )

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
        if bin_lower_type != self.EOS_CODE and bin_upper_type != self.EOS_CODE:
            if self.penalizer.to_penalize(bin_lower_type, bin_upper_type, resource_type):
                reward = self.misplace_penalty_factor * self.reward_per_level[request_type]
            else:
                reward = self.correct_place_factor * self.reward_per_level[request_type]
        
        # If placed premium request at EOS while there were available nodes
        # Give negative reward
        if request_type == 1 and bin_upper_type == self.EOS_CODE:
            if not np.all(feasible_mask[1:] == 1):
                reward = self.premium_rejected

        # If free request is placed at EOS
        if request_type == 0 and bin_upper_type == self.EOS_CODE:
            reward = 0

        return reward

    def compute_reward_batch(self,
                       batch,
                       total_num_nodes,
                       bins,
                       resources,
                       feasible_mask,
                       penalties,
                       is_eos_bin
                       ):

        batch_size = batch.shape[0]
        num_elements = batch.shape[1]
        num_features = batch.shape[2]
        batch_indices = tf.range(batch_size, dtype='int32')
        all_bins = batch[:, :total_num_nodes]
        all_resources = batch[:, total_num_nodes:]

        bins_lower_type = bins[:, 3]
        bins_upper_type = bins[:, 4]
        bins_remaining_resources = bins[:, :3]

        resources_types = resources[:, 3]
        users_type = tf.cast(resources[:, 4], dtype="int32")
        resources_demands = resources[:, :3]

        tiled_factors = tf.tile(self.mis_placement_tensor, [batch_size, 1], ).numpy()
        tiled_rewards_per_level = tf.tile(self.reward_per_level_tensor, [batch_size, 1], ).numpy()

        factors = tiled_factors[batch_indices, penalties]
        
        base_rewards = factors * tiled_rewards_per_level[batch_indices, users_type]
        
        # Set reward to ZERO for EOS nodes
        # If FREE REQUEST is placed at EOS reward will be ZERO
        base_rewards = base_rewards * (1 - is_eos_bin)

        # Marked as 1 = All full
        # Marked as 0 = NOT all full
        are_bins_full = bins_full_checker(feasible_mask, num_elements)

        # If placed PREMIUM REQUEST at EOS while there were available nodes
        # Give negative reward
        # Marked as 1 = Premium and Full
        # Marked as 0 = Not Full or Not Premium
        is_premium_and_bins_are_full = is_premium_wrongly_rejected_checker(
            are_bins_full, users_type, is_eos_bin
        )

        reward_premium_and_bins_are_full = is_premium_and_bins_are_full * self.premium_rejected

        # Final reward
        reward = base_rewards + reward_premium_and_bins_are_full
        
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
        self.premium_rejected = opts['premium_rejected']
        self.free_rejected = opts['free_rejected']

        self.mis_placement_tensor = tf.constant([
            self.correct_place_factor,
            self.misplace_penalty_factor,
        ],
            dtype='float32',
            shape=(1, 2)
        )

        self.reward_per_level_tensor = tf.constant(
            self.reward_per_level,
            dtype='float32',
            shape=(1,2)
        )

        self.rejection_tensor = tf.constant([
            self.premium_rejected,
            self.free_rejected
        ],
            dtype='float32',
            shape=(1, 2)
        )

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
        if bin_lower_type != self.EOS_CODE and bin_upper_type != self.EOS_CODE:

            skewness_reward = 0
            if bin_lower_type != self.EOS_CODE and bin_upper_type != self.EOS_CODE:

                bin_resource_variance = tfp.stats.variance(bin[:3])
        
                if bin_resource_variance == 0:
                    bin_resource_variance = 1
        
                skewness_reward = math.log(bin_resource_variance, 0.5) / 10

            if self.penalizer.to_penalize(bin_lower_type, bin_upper_type, resource_type):
                reward = self.misplace_penalty_factor * (self.reward_per_level[request_type] + skewness_reward)
            else:
                reward = self.correct_place_factor * ( self.reward_per_level[request_type] + skewness_reward)
        
        # If placed premium request at EOS while there were available nodes
        # Give negative reward
        if request_type == 1 and bin_upper_type == self.EOS_CODE:
            if not np.all(feasible_mask[1:] == 1):
                reward = self.premium_rejected

        # If free request is placed at EOS
        if request_type == 0 and bin_upper_type == self.EOS_CODE:
            reward = 0

        return reward
    
    def compute_reward_batch(self,
                       batch,
                       total_num_nodes,
                       bins,
                       resources,
                       feasible_mask
                       ):

        batch_size = batch.shape[0]
        num_elements = batch.shape[1]
        num_features = batch.shape[2]
        batch_indices = tf.range(batch_size, dtype='int32')
        all_bins = batch[:, :total_num_nodes]
        all_resources = batch[:, total_num_nodes:]

        bins_lower_type = bins[:, 3]
        bins_upper_type = bins[:, 4]
        bins_remaining_resources = bins[:, :3]

        resources_types = resources[:, 3]
        users_type = tf.cast(resources[:, 4], dtype="int32")
        resources_demands = resources[:, :3]

        # Check if selected bins are EOS
        # Marked as 1 = EOS node
        # Marked as 0 = not a EOS node
        is_eos_bin = bins_eos_checker(bins, self.EOS_CODE, num_features)

        # Check for the penalty
        # Marked as 1 = there's penalty involved
        # Marked as 0 = no penalty
        penalties = self.penalizer.to_penalize_batch(
            bins_lower_type,
            bins_upper_type,
            resources_types
        )

        tiled_factors = tf.tile(self.mis_placement_tensor, [batch_size, 1], ).numpy()
        tiled_rewards_per_level = tf.tile(self.reward_per_level_tensor, [batch_size, 1], ).numpy()

        factors = tiled_factors[batch_indices, penalties]
        
        base_rewards = factors * tiled_rewards_per_level[batch_indices, users_type]
        
        # Set reward to ZERO for EOS nodes
        # If FREE REQUEST is placed at EOS reward will be ZERO
        base_rewards = base_rewards * (1 - is_eos_bin)

        # Marked as 1 = All full
        # Marked as 0 = NOT all full
        are_bins_full = bins_full_checker(feasible_mask, num_elements)

        # If placed PREMIUM REQUEST at EOS while there were available nodes
        # Give negative reward
        # Marked as 1 = Premium and Full
        # Marked as 0 = Not Full or Not Premium
        is_premium_and_bins_are_full = is_premium_wrongly_rejected_checker(
            are_bins_full, users_type, is_eos_bin
        )

        reward_premium_and_bins_are_full = is_premium_and_bins_are_full * self.premium_rejected

        # Final reward
        reward = base_rewards + reward_premium_and_bins_are_full
        
        return reward, penalties, is_eos_bin