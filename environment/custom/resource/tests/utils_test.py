import numpy as np
import unittest
import sys

sys.path.append('.')

# Custom Imports
from environment.custom.resource.utils import bins_full_checker, bins_eos_checker, is_premium_wrongly_rejected_checker


class TestUtils(unittest.TestCase):

    def setUp(self) -> None:
        self.EOS_CODE = 0
        

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

        actual_result = bins_eos_checker(bins, self.EOS_CODE, num_features)

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