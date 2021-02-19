import numpy as np
import unittest
import sys

sys.path.append('.')

from environment.custom.resource_v3.reward import GreedyReward, SingleNodeDominantReward, GlobalDominantReward, gini_calculator

class Greedy_0_1_Reward(unittest.TestCase):
    def test_greedy_0_1_reward(self):
        opts = {}
        EOS_CODE = -2
        EOS_NODE = np.full((1, 3), EOS_CODE, dtype='float32')
        rewarder = GreedyReward(opts, EOS_NODE)

        og_batch = np.array([
            [
                [ -2.,  -2.,  -2.], # EOS       (index 0)
                [100., 200., 300.], # Node 0    (index 1)
                [400., 500., 600.], # Node 1    (index 2)
                [ 10.,  20.,  30.], # Req 0     (index 3)
                [ 40.,  50.,  60.], # Req 1     (index 4)
            ],
            [
                [ -2.,  -2.,  -2.], # EOS       (index 0)
                [100., 200., 300.], # Node 0    (index 1)
                [400., 500., 600.], # Node 1    (index 2)
                [ 10.,  20.,  30.], # Req 0     (index 3)
                [ 40.,  50.,  60.], # Req 1     (index 4)
            ]], dtype='float32')
        
        num_nodes = 3
        batch_indices = [0, 1]
        selected_node_ids = [0, 2]
        selected_resource_ids = [3, 4]
        feasible_mask = None

        # Pick the nodes and reqs
        selected_nodes = og_batch[batch_indices, selected_node_ids]
        selected_reqs = og_batch[batch_indices, selected_resource_ids]

        # Compute remaining resources
        remaining_resources_nodes = selected_nodes - selected_reqs

        # Update the batch
        updated_batch = og_batch.copy()
        updated_batch[batch_indices, selected_node_ids] = remaining_resources_nodes

        actual_rewards = rewarder.compute_reward(
            updated_batch,
            og_batch,
            num_nodes,
            selected_nodes,
            selected_reqs,
            feasible_mask
        )

        expected_rewards = np.array([
            0, # Placed at EOS  -> no reward
            1, # Placed at node -> receive 1
        ], dtype="float32")

        self.assertEqual(
            actual_rewards.numpy().tolist(),
            expected_rewards.tolist()
        )

class SingleNodeDominantTest(unittest.TestCase):
    def test_fair_single_node_dominant_reward(self):
        opts = {
            "rejection_penalty": -1
        }
        EOS_CODE = -2
        EOS_NODE = np.full((1, 3), EOS_CODE, dtype='float32')
        rewarder = SingleNodeDominantReward(opts, EOS_NODE)

        og_batch = np.array([
            [
                [ -2.,  -2.,  -2.], # EOS       (index 0)
                [100., 200., 300.], # Node 0    (index 1)
                [400., 500., 600.], # Node 1    (index 2)
                [ 10.,  20.,  30.], # Req 0     (index 3)
                [ 40.,  50.,  60.], # Req 1     (index 4)
            ],
            [
                [ -2.,  -2.,  -2.], # EOS       (index 0)
                [100., 200., 300.], # Node 0    (index 1)
                [400., 500., 600.], # Node 1    (index 2)
                [ 10.,  20.,  30.], # Req 0     (index 3)
                [ 40.,  50.,  60.], # Req 1     (index 4)
            ]], dtype='float32')
        
        num_nodes = 3
        batch_indices = [0, 1]
        selected_node_ids = [0, 2]
        selected_resource_ids = [3, 4]
        feasible_mask = None

        # Pick the nodes and reqs
        selected_nodes = og_batch[batch_indices, selected_node_ids]
        selected_reqs = og_batch[batch_indices, selected_resource_ids]

        # Compute remaining resources
        remaining_resources_nodes = selected_nodes - selected_reqs

        # Update the batch
        updated_batch = og_batch.copy()
        updated_batch[batch_indices, selected_node_ids] = remaining_resources_nodes

        actual_rewards = rewarder.compute_reward(
            updated_batch,
            og_batch,
            num_nodes,
            selected_nodes,
            selected_reqs,
            feasible_mask
        )

        expected_rewards = np.array([
            -1,  # Placed at EOS  -> no reward
            360, # Placed at node -> receive dominant resource of the node
        ], dtype="float32")

        self.assertEqual(
            actual_rewards.numpy().tolist(),
            expected_rewards.tolist()
        )

class GlobalDominantTest(unittest.TestCase):
    def test_fair_single_node_dominant_reward(self):
        opts = {
            "rejection_penalty": -1
        }
        EOS_CODE = -2
        EOS_NODE = np.full((1, 3), EOS_CODE, dtype='float32')
        rewarder = GlobalDominantReward(opts, EOS_NODE)

        og_batch = np.array([
            [
                [ -2.,  -2.,  -2.], # EOS       (index 0)
                [100., 200., 300.], # Node 0    (index 1)
                [400., 500., 600.], # Node 1    (index 2)
                [ 10.,  20.,  30.], # Req 0     (index 3)
                [ 40.,  50.,  60.], # Req 1     (index 4)
            ],
            [
                [ -2.,  -2.,  -2.], # EOS       (index 0)
                [100., 200., 300.], # Node 0    (index 1)
                [400., 500., 600.], # Node 1    (index 2)
                [ 10.,  20.,  30.], # Req 0     (index 3)
                [ 40.,  50.,  60.], # Req 1     (index 4)
            ]], dtype='float32')
        
        num_nodes = 3
        batch_indices = [0, 1]
        selected_node_ids = [0, 2]
        selected_resource_ids = [3, 4]
        feasible_mask = None

        # Pick the nodes and reqs
        selected_nodes = og_batch[batch_indices, selected_node_ids]
        selected_reqs = og_batch[batch_indices, selected_resource_ids]

        # Compute remaining resources
        remaining_resources_nodes = selected_nodes - selected_reqs

        # Update the batch
        updated_batch = og_batch.copy()
        updated_batch[batch_indices, selected_node_ids] = remaining_resources_nodes

        actual_rewards = rewarder.compute_reward(
            updated_batch,
            og_batch,
            num_nodes,
            selected_nodes,
            selected_reqs,
            feasible_mask
        )

        expected_rewards = np.array([
            -1,  # Placed at EOS  -> no reward
            100, # Placed at node -> receive dominant resource of all nodes
        ], dtype="float32")

        self.assertEqual(
            actual_rewards.numpy().tolist(),
            expected_rewards.tolist()
        )


class TestGiniCalculartor(unittest.TestCase):
    def test_gini_calculator(self):
        entries = np.array([
            [15, 15, 30, 40],
            [10, 20, 35, 35],
            [0, 0, 0, 1],
            [1, 1, 1, 1],
            [0.4, 0.7, 0.1, 0.3],
            # [-10, -20, 35, 35], # Gini don't work with negatives
        ], dtype='float32')

        num_nodes = 4

        expected = np.array([
            0.225,
            0.225,
            0.750,
            0.000,
            0.317,
            # 0.500 # Gini don't work with negatives
        ], dtype='float32')

        actual = gini_calculator(entries, num_nodes)

        np.testing.assert_array_almost_equal(
            actual.numpy(),
            expected,
            decimal=3
        )