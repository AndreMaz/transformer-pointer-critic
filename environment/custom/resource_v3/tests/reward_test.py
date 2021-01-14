import numpy as np
import unittest
import sys

sys.path.append('.')

from environment.custom.resource_v3.reward import gini_calculator

class TestGiniCalculartor(unittest.TestCase):
    def test_gini_calculator(self):
        entries = np.array([
            [15, 15, 30, 40],
            [10, 20, 35, 35],
            [0, 0, 0, 1],
            [1, 1, 1, 1],
            [0.4, 0.7, 0.1, 0.3],
            [-10, -20, 35, 35],
        ], dtype='float32')

        num_nodes = 4

        expected = np.array([
            0.225,
            0.225,
            0.750,
            0.000,
            0.317,
            0.500
        ], dtype='float32')

        actual = gini_calculator(entries, num_nodes)

        np.testing.assert_array_almost_equal(
            actual.numpy(),
            expected,
            decimal=3
        )