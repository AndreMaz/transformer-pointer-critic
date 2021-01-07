import tensorflow as tf
import numpy as np

def RewardFactory(opts: dict):
    rewards = {
        "fair": FairReward,
        "fair_v2": FairRewardV2
    }

    try:
        rewardType = opts['type']
        R = rewards[f'{rewardType}']
        print(f'"{rewardType.upper()}" reward selected.')
        return R(opts[f'{rewardType}'])
    except KeyError:
        raise NameError(f'Unknown Reward Name! Select one of {list(rewards.keys())}')

class FairReward():
    def __init__(self, opts: dict):
        super(FairReward, self).__init__()

    def compute_reward(self,
                       updated_batch,
                       original_batch,
                       total_num_nodes,
                       nodes,
                       reqs,
                       feasible_mask
                       ):

        all_bins = updated_batch[:, :total_num_nodes]
        bins_cpu = all_bins[:, :, 0]
        bins_ram = all_bins[:, :, 1]
        bins_mem = all_bins[:, :, 2]

        min_cpu = tf.math.reduce_min(bins_cpu, axis=1)
        min_ram = tf.math.reduce_min(bins_ram, axis=1)
        min_mem = tf.math.reduce_min(bins_mem, axis=1)

        min_vals = tf.convert_to_tensor([min_cpu, min_ram, min_mem])

        min_resource = tf.math.reduce_min(min_vals, axis=0)

        return min_resource

class FairRewardV2():
    def __init__(self, opts: dict):
        super(FairReward, self).__init__()

    def compute_reward(self,
                       updated_batch,
                       original_batch,
                       total_num_nodes,
                       nodes,
                       reqs,
                       feasible_mask
                       ):

        all_bins = updated_batch[:, :total_num_nodes]
        bins_cpu = all_bins[:, :, 0]
        bins_ram = all_bins[:, :, 1]
        bins_mem = all_bins[:, :, 2]

        min_cpu = tf.math.reduce_min(bins_cpu, axis=1)
        min_ram = tf.math.reduce_min(bins_ram, axis=1)
        min_mem = tf.math.reduce_min(bins_mem, axis=1)

        min_vals = tf.convert_to_tensor([min_cpu, min_ram, min_mem])

        min_resource = tf.math.reduce_min(min_vals, axis=0)

        return min_resource

if __name__ == "__main__":
    reward = FairReward({})

    batch = np.array([
        [
            [1, 2, 3],
            [5, 2, 6],
            [0, 0, 0],
            [0, 0, 0],
        ],
        [
            [5, 3, 7],
            [2, 1, 8],
            [0, 0, 0],
            [0, 0, 0],
        ]
    ], dtype='float32')

    total_num_nodes = 2

    nodes = None
    reqs = None
    feasible_mask  = None

    reward.compute_reward(
                       batch,
                       total_num_nodes,
                       nodes,
                       reqs,
                       feasible_mask
                       )