import numpy as np
import unittest
import sys

from tensorflow.python.keras.backend import dtype


sys.path.append('.')

# Custom Imports
from environment.custom.resource.reward import GreedyReward, bins_full_checker, bins_eos_checker, is_premium_wrongly_rejected_checker
from environment.custom.resource.penalty import GreedyPenalty

class TestItem(unittest.TestCase):

    def setUp(self) -> None:
        opts = {
                "CPU_misplace_penalty": 5,
                "RAM_misplace_penalty": 10,
                "MEM_misplace_penalty": 15
        }

        EOS_CODE = 0
        resource_normalization_factor = 1

        penalizer = GreedyPenalty(
            opts, EOS_CODE, resource_normalization_factor    
        )

        opts = {
            "reward_per_level": [ 10, 20 ],
            "misplace_penalty_factor": 0.5,
            "correct_place_factor": 1,
            "premium_rejected": -20,
            "free_rejected": 0
        }

        self.rewarder = GreedyReward(
            opts, penalizer, EOS_CODE
        )

    def test_constructor(self):
        expected_reward_per_level = [10, 20]
        misplace_penalty_factor = 0.5
        self.assertEqual(
            self.rewarder.reward_per_level, expected_reward_per_level
        )

        self.assertEqual(
            self.rewarder.misplace_penalty_factor, misplace_penalty_factor
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
        bin = [400., 500., 600.,   1.,   3.]
        # Premium user: 1
        # Request Type: 2
        resource = [40.,  50.,  60.,   2.,   1.]

        expected_reward = 20

        feasible_mask = np.array([
            [0.,   0.,   0.,   1.,   1.],
        ],dtype='float32')

        actual_reward = self.rewarder.compute_reward(
            batch, total_num_nodes, bin, resource, feasible_mask
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
        bin = [400., 500., 600.,   1.,   3.]
        # Premium user: 1
        # Request Type: 2
        resource = [40.,  50.,  60.,   2.,   1.]

        expected_reward = 20

        feasible_bin_mask = np.array([
            [ 0.,  0.,  0.,   1.,   1.],
        ], dtype='float32') 

        actual_reward = self.rewarder.compute_reward(
            batch, total_num_nodes, bin, resource, feasible_bin_mask
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
        bin = [400., 500., 600.,   1.,   3.]
        # Premium user: 0
        # Request Type: 5
        resource = [40.,  50.,  60.,   2.,   0.]

        feasible_bin_mask = np.array([
            [ 0.,  0.,  0.,   1.,   1.],
        ], dtype='float32')

        expected_reward = 10
    
        actual_reward = self.rewarder.compute_reward(
            batch, total_num_nodes, bin, resource, feasible_bin_mask
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
        bin = [400., 500., 600.,   1.,   3.]
        # Premium user: 0
        # Request Type: 5
        resource = [40.,  50.,  60.,   5.,   0.]

        feasible_bin_mask = np.array([
            [ 0.,  0.,  0.,   1.,   1.],
        ], dtype='float32')

        expected_reward = 5

        actual_reward = self.rewarder.compute_reward(
            batch, total_num_nodes, bin, resource, feasible_bin_mask
        )

        self.assertEqual(actual_reward, expected_reward)

############################################################
##################  COPY PASTA OF TESTS ABOVE   ############
############################################################

    def test_reward_batch_UNpenalized(self):
        batch = np.array([[
            [0.,   0.,   0.,   0.,   0.],
            [100., 200., 300.,   0.,   2.],
            [400., 500., 600.,   0.,   3.],
            [10.,  20.,  30.,   1.,   1.],
            [40.,  50.,  60.,   8.,   1.]]],
            dtype='float32')

        total_num_nodes = 3

        # Lower bound type: 0
        # Upper bound type: 3
        bin = np.array([[400., 500., 600.,   0.,   3.]], dtype='float32')
        # Premium user: 1
        # Request Type: 2
        resource = np.array([[40.,  50.,  60.,   2.,   1.]], dtype='float32')

        expected_reward = [20]

        feasible_mask = np.array([
            [0.,   0.,   0.,   1.,   1.],
        ],dtype='float32')

        actual_reward, _, _ = self.rewarder.compute_reward_batch(
            batch, total_num_nodes, bin, resource, feasible_mask
        )

        self.assertEqual(actual_reward.numpy().tolist(), expected_reward)
    
    def test_reward_batch_UNpenalized_premium(self):
        batch = np.array([[
            [0.,   0.,   0.,   0.,   0.],
            [100., 200., 300.,   0.,   2.],
            [400., 500., 600.,   0.,   3.],
            [10.,  20.,  30.,   1.,   1.],
            [40.,  50.,  60.,   8.,   1.]]],
            dtype='float32')

        total_num_nodes = 3

        # Lower bound type: 0
        # Upper bound type: 3
        bin = np.array([[400., 500., 600.,   0.,   3.]], dtype='float32')
        # Premium user: 1
        # Request Type: 2
        resource = np.array([[40.,  50.,  60.,   2.,   1.]], dtype='float32')

        expected_reward = [20]

        feasible_bin_mask = np.array([
            [ 0.,  0.,  0.,   1.,   1.],
        ], dtype='float32') 

        actual_reward, _, _ = self.rewarder.compute_reward_batch(
            batch, total_num_nodes, bin, resource, feasible_bin_mask
        )

        self.assertEqual(actual_reward.numpy().tolist(), expected_reward)
    
    def test_reward_batch_UNpenalized_free(self):
        batch = np.array([[
            [0.,   0.,   0.,   0.,   0.],
            [100., 200., 300.,   0.,   2.],
            [400., 500., 600.,   0.,   3.],
            [10.,  20.,  30.,   1.,   1.],
            [40.,  50.,  60.,   8.,   1.]]],
            dtype='float32')

        total_num_nodes = 3

        # Lower bound type: 0
        # Upper bound type: 3
        bin = np.array([[400., 500., 600.,   0.,   3.]], dtype='float32')
        # Premium user: 0
        # Request Type: 5
        resource = np.array([[40.,  50.,  60.,   2.,   0.]], dtype='float32')

        feasible_bin_mask = np.array([
            [ 0.,  0.,  0.,   1.,   1.],
        ], dtype='float32')

        expected_reward = [10]
    
        actual_reward, _, _ = self.rewarder.compute_reward_batch(
            batch, total_num_nodes, bin, resource, feasible_bin_mask
        )

        self.assertEqual(actual_reward.numpy().tolist(), expected_reward)

    def test_reward_batch_penalized_free(self):
        batch = np.array([[
            [0.,   0.,   0.,   0.,   0.],
            [100., 200., 300.,   0.,   2.],
            [400., 500., 600.,   0.,   3.],
            [10.,  20.,  30.,   1.,   1.],
            [40.,  50.,  60.,   8.,   1.]]],
            dtype='float32')

        total_num_nodes = 3

        # Lower bound type: 0
        # Upper bound type: 3
        bin = np.array([[400., 500., 600.,   0.,   3.]], dtype='float32')
        # Premium user: 0
        # Request Type: 5
        resource = np.array([[40.,  50.,  60.,   5.,   0.]], dtype='float32')

        feasible_bin_mask = np.array([
            [ 0.,  0.,  0.,   1.,   1.],
        ], dtype='float32')

        expected_reward = [5]

        actual_reward, _, _ = self.rewarder.compute_reward_batch(
            batch, total_num_nodes, bin, resource, feasible_bin_mask
        )

        self.assertEqual(actual_reward.numpy().tolist(), expected_reward)

    ##################################### NEW TESTS #####################################

    def test_reward_batch_premium_first_UNpenalized_second_penalized(self):
        batch = np.array([
            [
                [0.,   0.,   0.,   0.,   0.],
                [100., 200., 300.,   0.,   2.],
                [400., 500., 600.,   0.,   3.],
                [10.,  20.,  30.,   1.,   1.],
                [40.,  50.,  60.,   8.,   1.]
            ],
            [
                [0.,   0.,   0.,   0.,   0.],
                [100., 200., 300.,   0.,   2.],
                [400., 500., 600.,   0.,   3.],
                [10.,  20.,  30.,   1.,   1.],
                [40.,  50.,  60.,   8.,   1.]
            ]
        ],dtype='float32')

        total_num_nodes = 3

        # Lower bound type: 0
        # Upper bound type: 3
        bin = np.array([
            [400., 500., 600.,   0.,   3.], # No penalty
            [400., 500., 600.,   0.,   3.], # Will receive penalty
        ], dtype='float32')
        # Premium user: 0
        # Request Type: 5
        resource = np.array([
            [40.,  50.,  60.,   1.,   1.], # Withing the range
            [40.,  50.,  60.,   5.,   1.]  # Outside the range
        ], dtype='float32')

        feasible_bin_mask = np.array([
            [ 0.,  0.,  0.,   1.,   1.],
            [ 0.,  0.,  0.,   1.,   1.],
        ], dtype='float32')

        expected_reward = [20, 10]

        actual_reward, _, _ = self.rewarder.compute_reward_batch(
            batch, total_num_nodes, bin, resource, feasible_bin_mask
        )

        self.assertEqual(actual_reward.numpy().tolist(), expected_reward)

    def test_reward_batch_premium_first_at_EOS_wrong_second_at_EOS_correct(self):
        batch = np.array([
            [
                [0.,   0.,   0.,   0.,   0.],
                [100., 200., 300.,   0.,   2.],
                [400., 500., 600.,   0.,   3.],
                [10.,  20.,  30.,   1.,   1.],
                [40.,  50.,  60.,   8.,   1.]
            ],
            [
                [0.,   0.,   0.,   0.,   0.],
                [100., 200., 300.,   0.,   2.],
                [400., 500., 600.,   0.,   3.],
                [10.,  20.,  30.,   1.,   1.],
                [40.,  50.,  60.,   8.,   1.]
            ]
        ],dtype='float32')

        total_num_nodes = 3

        # Lower bound type: 0
        # Upper bound type: 3
        bin = np.array([
            [0.,   0.,   0.,   0.,   0.], # EOS node
            [0.,   0.,   0.,   0.,   0.], # EOS node
        ], dtype='float32')
        # Premium user: 0
        # Request Type: 5
        resource = np.array([
            [40.,  50.,  60.,   1.,   1.],
            [40.,  50.,  60.,   5.,   1.]
        ], dtype='float32')

        feasible_bin_mask = np.array([
            [ 0.,  0.,  0.,   1.,   1.], # Give negative because there are nodes available
            [ 0.,  1.,  1.,   1.,   1.], # Only EOS is available
        ], dtype='float32')

        expected_reward = [-20, 0]

        actual_reward, _, _ = self.rewarder.compute_reward_batch(
            batch, total_num_nodes, bin, resource, feasible_bin_mask
        )

        self.assertEqual(actual_reward.numpy().tolist(), expected_reward)
    
    def test_complex(self):
        batch = np.array([
            [
                [0.,   0.,   0.,   0.,   0.],
                [100., 200., 300.,   0.,   2.],
                [400., 500., 600.,   0.,   3.],
                [10.,  20.,  30.,   1.,   1.],
                [40.,  50.,  60.,   8.,   0.]
            ],
            [
                [0.,   0.,   0.,   0.,   0.],
                [100., 200., 300.,   0.,   2.],
                [400., 500., 600.,   0.,   3.],
                [10.,  20.,  30.,   1.,   1.],
                [40.,  50.,  60.,   8.,   0.]
            ],
            [
                [0.,   0.,   0.,   0.,   0.],
                [100., 200., 300.,   0.,   2.],
                [400., 500., 600.,   0.,   3.],
                [10.,  20.,  30.,   1.,   1.],
                [40.,  50.,  60.,   8.,   0.]
            ],
            [
                [0.,   0.,   0.,   0.,   0.],
                [100., 200., 300.,   0.,   2.],
                [400., 500., 600.,   0.,   3.],
                [10.,  20.,  30.,   1.,   1.],
                [40.,  50.,  60.,   8.,   0.]
            ]
        ],dtype='float32')

        total_num_nodes = 3

        # Lower bound type: 0
        # Upper bound type: 3
        bin = np.array([
            [100., 200., 300.,   0.,   2.],
            [  0.,   0.,   0.,   0.,   0.], # EOS node
            [100., 200., 300.,   0.,   2.],
            [  0.,   0.,   0.,   0.,   0.], # EOS node
        ], dtype='float32')
        # Premium user: 0
        # Request Type: 5
        resource = np.array([
            [10.,  20.,  30.,   1.,   1.],
            [40.,  50.,  60.,   8.,   1.],
            [10.,  20.,  30.,   1.,   0.],
            [40.,  50.,  60.,   8.,   0.]
        ], dtype='float32')

        feasible_bin_mask = np.array([
            [ 0.,  0.,  0.,   1.,   1.], # Give negative because there are nodes available
            [ 0.,  1.,  1.,   1.,   1.], # Only EOS is available
            [ 0.,  0.,  0.,   1.,   1.], # Give negative because there are nodes available
            [ 0.,  1.,  1.,   1.,   1.], # Only EOS is available
        ], dtype='float32')

        expected_reward = [20, 0, 10, 0]

        actual_reward, _, _ = self.rewarder.compute_reward_batch(
            batch, total_num_nodes, bin, resource, feasible_bin_mask
        )

        self.assertEqual(actual_reward.numpy().tolist(), expected_reward)
    

    def test_bins_full_checker(self):
        feasible_bin_mask = np.array([
            [ 0.,  0.,  0.,   1.,   1.],
            [ 0.,  0.,  1.,   1.,   1.],
            [ 0.,  1.,  1.,   1.,   1.],
        ], dtype='float32')

        num_features = 5
        excepted_result = [0, 0, 1]

        actual_result = bins_full_checker(feasible_bin_mask, num_features)

        self.assertEqual(actual_result.numpy().tolist(), excepted_result)

    def test_bins_full_checker(self):
        bins = np.array([
            [ 10.,  20.,  30.,   2.,   5.],
            [ 0.,  0.,  0.,   1.,   1.],
            [ 0.,  10.,  10.,   1.,   1.],
            [ 0.,  0.,  0.,   0.,   0.], # Only this one is EOS
        ], dtype='float32')

        num_features = 5
        excepted_result = [0, 0, 0, 1]

        actual_result = bins_eos_checker(bins, self.rewarder.EOS_CODE, num_features)

        self.assertEqual(actual_result.numpy().tolist(), excepted_result)

    def test_is_premium_and_bins_are_full_checker(self):
        feasible_bin_mask = np.array([
            [ 0.,  1.,  1.,   1.,   1.],
            [ 0.,  0.,  0.,   1.,   1.],
            [ 0.,  1.,  1.,   1.,   1.], # All Full
            [ 0.,  0.,  0.,   1.,   1.],
        ], dtype='float32')

        num_features = 5

        user_types = np.array([
            0, # Free
            0, # Free
            1, # Premium
            1  # Premium
        ], dtype='float32')

        is_eos_bin = np.array([
            0,
            1,
            0,
            1
        ],dtype='int32')

        are_bins_full = bins_full_checker(feasible_bin_mask, num_features)

        expected_result = [
            0,
            0,
            0,
            1  # Rejected while there were space
        ]
        actual_result = is_premium_wrongly_rejected_checker(are_bins_full, user_types, is_eos_bin)

        self.assertEqual(actual_result.numpy().tolist(), expected_result)