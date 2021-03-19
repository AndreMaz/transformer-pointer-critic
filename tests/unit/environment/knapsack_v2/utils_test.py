import sys
sys.path.append('.')
import unittest
import numpy as np


from environment.custom.knapsack_v2.misc.utils import compute_remaining_resources


class TestUtils(unittest.TestCase):
    def test_round_ops(self):
        bins = np.array([
            [0.9, 0.0],
            [0.8, 0.6],
            [0.5, 0.1],
        ], dtype="float32")

        items = np.array([
            [0.7, 0.1],
            [0.1, 0.6],
            [0.4, 0.1],
        ], dtype="float32")

        expected_updated_bins = np.array([
            [0.9, 0.7],
            [0.8, 0.7],
            [0.5, 0.5],
        ], dtype="float32")

        actual_updated_bins = compute_remaining_resources(bins, items, 2)

        self.assertEqual(
            actual_updated_bins.tolist(),
            expected_updated_bins.tolist()
        )