import sys

from tensorflow.python.ops.gen_batch_ops import batch

sys.path.append('.')

import numpy as np
import unittest

# Custom Imports
from environment.custom.resource_v2.env import ResourceEnvironmentV2

class TestResource(unittest.TestCase):

    def setUp(self) -> None:
        ENV_CONFIG = {
            "description": "Environment configs.",

            "batch_size": 2,

            "num_features": 3,
            "num_profiles": 20,

            "profiles_sample_size": 2,
            "node_sample_size": 3,

            "req_min_val": 1,
            "req_max_val": 30,

            "node_min_val": 80,
            "node_max_val": 100,

            "reward": {
                "type": "fair_v2",
                "fair": {},
                "fair_v2": {},
                "gini": {}
            }
        }
        self.env = ResourceEnvironmentV2('ResourceV2', ENV_CONFIG)
        
    def test_constructor(self):
        self.assertEqual(self.env.name, 'ResourceV2')
        self.assertIsNotNone(self.env.rewarder)
        self.assertEqual(self.env.batch_size, 2)
        self.assertEqual(self.env.profiles_sample_size, 2)
        self.assertEqual(self.env.node_sample_size, 3)

    def test_shapes(self):
        self.assertEqual(self.env.total_profiles.shape, (20, 3))

        self.assertEqual(self.env.batch.shape, (2, 5, 3))
        self.assertEqual(self.env.bin_net_mask.shape, (2,5))
        self.assertEqual(self.env.resource_net_mask.shape, (2,5))
        self.assertEqual(self.env.mha_used_mask.shape, (2, 1, 1, 5))

    def test_initial_masks(self):
        expected_bin_mask = np.array([
            [0, 0, 0, 1, 1],
            [0, 0, 0, 1, 1],
        ], dtype='float32')

        expected_resource_mask = np.array([
            [1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0],
        ], dtype='float32')

        expected_mha_mask = np.array([
            [[[0, 0, 0, 0, 0]]],
            [[[0, 0, 0, 0, 0]]],
        ], dtype='float32')

        state, bin_net_mask, resource_net_mask, mha_mask = self.env.state()

        self.assertEqual(
            bin_net_mask.tolist(),
            expected_bin_mask.tolist()
        )

        self.assertEqual(
            resource_net_mask.tolist(),
            expected_resource_mask.tolist()
        )

        self.assertEqual(
            mha_mask.tolist(),
            expected_mha_mask.tolist()
        )
    
    def test_batch_values(self):

        state, _, _, _ = self.env.state()

        nodes = state[:, :self.env.node_sample_size]
        reqs = state[:, self.env.node_sample_size:]

        self.assertTrue(
            np.all(nodes >= 0.8)
        )

        self.assertTrue(
            np.all(reqs < 0.3)
        )
    def test_testing_mode(self):
        self.env.set_testing_mode(
            batch_size=2,
            node_sample_size=3,
            profiles_sample_size=2
        )
        self.env.reset()

        state, _, _, _ = self.env.state()

        nodes = state[:, :self.env.node_sample_size]
        reqs = state[:, self.env.node_sample_size:]

        self.assertEqual(len(self.env.history), 2)
        self.assertEqual(len(self.env.history[0]), 3)
        self.assertEqual(len(self.env.history[1]), 3)

        # Grab the values and shape them into a (batch, node, feature)
        b0_n0 = self.env.history[0][0].get_tensor_rep()
        b0_n1 = self.env.history[0][1].get_tensor_rep()
        b0_n2 = self.env.history[0][2].get_tensor_rep()
        
        batch0_nodes = np.array([b0_n0, b0_n1, b0_n2])
        
        b1_n0 = self.env.history[1][0].get_tensor_rep()
        b1_n1 = self.env.history[1][1].get_tensor_rep()
        b1_n2 = self.env.history[1][2].get_tensor_rep()
        
        batch1_nodes = np.array([b1_n0, b1_n1, b1_n2])

        batch_rep = np.array([batch0_nodes, batch1_nodes])

        self.assertEqual(
            nodes.tolist(),
            batch_rep.tolist()
        )