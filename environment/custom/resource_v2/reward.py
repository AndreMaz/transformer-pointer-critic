import tensorflow as tf
import numpy as np
from tensorflow.python.keras.engine import node

def RewardFactory(opts: dict):
    rewards = {
        "fair": FairReward,
        "fair_v2": FairRewardV2,
        "gini": GiniReward
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
        super(FairRewardV2, self).__init__()

    def compute_reward(self,
                       updated_batch,
                       original_batch,
                       total_num_nodes,
                       nodes,
                       reqs,
                       feasible_mask
                       ):

        batch_size = updated_batch.shape[0]

        # Find dominant resource for the selected nodes
        updated_nodes = nodes - reqs
        min_resource_selected_node = tf.math.reduce_min(updated_nodes, axis=1)

        # Find dominant resource IF the request were placed at other nodes
        # original_batch_nodes = original_batch[:, :total_num_nodes]
        # expanded_reqs = tf.tile(tf.expand_dims(reqs, 1), [1, total_num_nodes, 1])

        # diff_placement = original_batch_nodes - expanded_reqs

        # bins_cpu = diff_placement[:, :, 0]
        # bins_ram = diff_placement[:, :, 1]
        # bins_mem = diff_placement[:, :, 2]

        # min_cpu = tf.math.reduce_min(bins_cpu, axis=1)
        # min_ram = tf.math.reduce_min(bins_ram, axis=1)
        # min_mem = tf.math.reduce_min(bins_mem, axis=1)

        # min_vals = tf.convert_to_tensor([min_cpu, min_ram, min_mem])

        # min_resource_all_nodes = tf.math.reduce_min(min_vals, axis=0)

        return min_resource_selected_node

class GiniReward():
    def __init__(self, opts: dict):
        super(GiniReward, self).__init__()

    def compute_reward(self,
                       updated_batch,
                       original_batch,
                       total_num_nodes,
                       nodes,
                       reqs,
                       feasible_mask
                       ):

        batch_size = updated_batch.shape[0]

        all_bins = updated_batch[:, :total_num_nodes]
        bins_cpu = all_bins[:, :, 0]
        bins_ram = all_bins[:, :, 1]
        bins_mem = all_bins[:, :, 2]

        gini_cpu = 1 - gini_calculator(bins_cpu, total_num_nodes)
        gini_ram = 1 - gini_calculator(bins_ram, total_num_nodes)
        gini_mem = 1 - gini_calculator(bins_mem, total_num_nodes)

        return gini_cpu + gini_ram + gini_mem

# Links: https://goodcalculators.com/gini-coefficient-calculator/
# Links: https://shlegeris.com/gini.html
def gini_calculator(entries, num_nodes):
    
    batch_size = entries.shape[0]

    # Node indices from [1, num_nodes]
    node_indices =  tf.range(1, num_nodes+1, dtype='float32')
    
    # Sorted
    entries = tf.sort(entries, axis=-1, direction='ASCENDING')
    
    # Positive entries
    # Set negative values to 0
    # positives = tf.cast(tf.greater(entries, 0), dtype='float32')
    # entries = entries * positives

    numerator = 2*tf.reduce_sum(
        (num_nodes + 1 - node_indices)*entries,
        axis=-1
    )
    denominator = num_nodes * tf.reduce_sum(entries, axis=-1)

    gini = (num_nodes + 1)/ num_nodes - (numerator / denominator)
    
    return gini


if __name__ == "__main__":
    entries = np.array([
        [15, 15, 30, 40],
        [10, 20, 35, 35],
        [-10, -20, 35, 35],
        [-0.01, 0, 0, 0],
        [0, 0, 0, 0],
    ], dtype='float32')

    num_nodes = 4

    gini_calculator(entries, num_nodes)

    # reward = FairRewardV2({})

    # updated_batch = np.array([
    #     [
    #         [10, 20, 30],
    #         [50, 20, 60],
    #         [0, 0, 0],
    #         [0, 0, 0],
    #     ],
    #     [
    #         [5, 3, 7],
    #         [2, 1, 8],
    #         [0, 0, 0],
    #         [0, 0, 0],
    #     ]
    # ], dtype='float32')
    
    # original_batch = np.array([
    #     [
    #         [1, 2, 3],
    #         [5, 2, 6],
    #         [0, 0, 0],
    #         [0, 0, 0],
    #     ],
    #     [
    #         [5, 3, 7],
    #         [2, 1, 8],
    #         [0, 0, 0],
    #         [0, 0, 0],
    #     ]
    # ], dtype='float32')

    # total_num_nodes = 2

    # nodes = None
    # reqs = None
    # feasible_mask  = None

    # reward.compute_reward(
    #                    updated_batch,
    #                    original_batch,
    #                    total_num_nodes,
    #                    nodes,
    #                    reqs,
    #                    feasible_mask
    #                    )