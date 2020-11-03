import numpy as np
import unittest
import sys
sys.path.append('.')

# Custom Imports
from environment.custom.resource.reward import GreedyReward
from environment.custom.resource.penalty import GreedyPenalty

class TestItem(unittest.TestCase):

    def setUp(self) -> None:
        opts = {
                "CPU_misplace_penalty": 5,
                "RAM_misplace_penalty": 10,
                "MEM_misplace_penalty": 15
        }

        EOS_CODE = -1
        resource_normalization_factor = 1

        penalizer = GreedyPenalty(
            opts, EOS_CODE, resource_normalization_factor    
        )

        opts = {
            "reward_per_level": [ 10, 20 ],
            "misplace_reward_penalty": 5
        }

        self.rewarder = GreedyReward(
            opts, penalizer
        )

    def test_constructor(self):
        expected_reward_per_level = [10, 20]
        expected_misplace_penalty = 5
        self.assertEqual(
            self.rewarder.reward_per_level, expected_reward_per_level
        )

        self.assertEqual(
            self.rewarder.misplace_reward_penalty, expected_misplace_penalty
        )

        self.assertIsNotNone(self.rewarder.penalizer)

    def test_reward_UNpenalized(self):
        batch = np.array([
            [0.,   0.,   0.,   0.,   0.],
            [100., 200., 300.,   0.,   2.],
            [400., 500., 600.,   0.,   3.],
            [10.,  20.,  30.,   1.,   1.],
            [40.,  50.,  60.,   8.,   1.]],
            dtype='float32')

        total_num_nodes = 3

        # Lower bound type: 0
        # Upper bound type: 3
        bin = [400., 500., 600.,   0.,   3.]
        # Premium user: 1
        # Request Type: 2
        resource = [40.,  50.,  60.,   2.,   1.]

        expected_reward = 20

        actual_reward = self.rewarder.compute_reward(
            batch, total_num_nodes, bin, resource
        )

        self.assertEqual(actual_reward, expected_reward)
    
    
    def test_reward_UNpenalized_premium(self):
        batch = np.array([
            [0.,   0.,   0.,   0.,   0.],
            [100., 200., 300.,   0.,   2.],
            [400., 500., 600.,   0.,   3.],
            [10.,  20.,  30.,   1.,   1.],
            [40.,  50.,  60.,   8.,   1.]],
            dtype='float32')

        total_num_nodes = 3

        # Lower bound type: 0
        # Upper bound type: 3
        bin = [400., 500., 600.,   0.,   3.]
        # Premium user: 1
        # Request Type: 2
        resource = [40.,  50.,  60.,   2.,   1.]

        expected_reward = 20

        actual_reward = self.rewarder.compute_reward(
            batch, total_num_nodes, bin, resource
        )

        self.assertEqual(actual_reward, expected_reward)
    
    def test_reward_UNpenalized_free(self):
        batch = np.array([
            [0.,   0.,   0.,   0.,   0.],
            [100., 200., 300.,   0.,   2.],
            [400., 500., 600.,   0.,   3.],
            [10.,  20.,  30.,   1.,   1.],
            [40.,  50.,  60.,   8.,   1.]],
            dtype='float32')

        total_num_nodes = 3

        # Lower bound type: 0
        # Upper bound type: 3
        bin = [400., 500., 600.,   0.,   3.]
        # Premium user: 0
        # Request Type: 5
        resource = [40.,  50.,  60.,   2.,   0.]

        expected_reward = 10

        actual_reward = self.rewarder.compute_reward(
            batch, total_num_nodes, bin, resource
        )

        self.assertEqual(actual_reward, expected_reward)

    def test_reward_penalized_free(self):
        batch = np.array([
            [0.,   0.,   0.,   0.,   0.],
            [100., 200., 300.,   0.,   2.],
            [400., 500., 600.,   0.,   3.],
            [10.,  20.,  30.,   1.,   1.],
            [40.,  50.,  60.,   8.,   1.]],
            dtype='float32')

        total_num_nodes = 3

        # Lower bound type: 0
        # Upper bound type: 3
        bin = [400., 500., 600.,   0.,   3.]
        # Premium user: 0
        # Request Type: 5
        resource = [40.,  50.,  60.,   5.,   0.]

        expected_reward = 5

        actual_reward = self.rewarder.compute_reward(
            batch, total_num_nodes, bin, resource
        )

        self.assertEqual(actual_reward, expected_reward)