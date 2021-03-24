import numpy as np
import unittest
import sys

sys.path.append('.')

from environment.custom.knapsack_v2.reward import GreedyReward
from environment.custom.knapsack_v2.misc.utils import compute_remaining_resources

class GreedyRewardTest(unittest.TestCase):
    def test_greedy_reward(self):
        opts = {}
        num_nodes = 3
        batch_size = 2

        EOS_CODE = -2
        EOS_NODE = np.full((1, 2), EOS_CODE, dtype='float32')
        rewarder = GreedyReward(opts, EOS_NODE)

        og_batch = np.array([[[-2., -2.],
                              [0.8,  0.0],
                              [0.8,  0.0],
                              [0.9,  0.0],
                              [0.1,  0.99],
                              [0.3,  0.3]],

                             [[-2., -2.],
                              [0.8,  0.0],
                              [0.9,  0.0],
                              [0.8,  0.0],
                              [0.2,  0.99],
                              [0.5, 0.75]]], dtype="float32")
        
        # num_nodes = 3
        batch_indices = [0, 1]
        selected_node_ids = [0, 2]
        selected_resource_ids = [4, 5]
        feasible_mask = None

        # Pick the nodes and reqs
        selected_nodes = og_batch[batch_indices, selected_node_ids]
        selected_reqs = og_batch[batch_indices, selected_resource_ids]

        # Update the batch
        updated_batch = og_batch.copy()

        actual_rewards = rewarder.compute_reward(
            updated_batch, # We don't care about the value in Greedy Reward
            og_batch, # We don't care about the value in Greedy Reward
            num_nodes,
            selected_nodes,
            selected_reqs,
            feasible_mask,
            selected_node_ids
        )

        expected_rewards = np.array([
            0.0, # Placed at EOS  -> no reward
            0.75, # Placed at node -> receive reward equal to item value
        ], dtype="float32")

        self.assertEqual(
            actual_rewards.numpy().tolist(),
            expected_rewards.tolist()
        )
