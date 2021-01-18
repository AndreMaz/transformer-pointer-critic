import sys
sys.path.append('.')
import unittest
import numpy as np

# Custom Imports
from environment.custom.resource_v3.env import ResourceEnvironmentV3

class TestResource(unittest.TestCase):

    def setUp(self) -> None:
        ENV_CONFIG = {
            "description": "Environment configs.",

            "batch_size": 2,

            "normalization_factor": 100,
            "decimal_precision": 2,

            "num_features": 3,
            "num_profiles": 20,

            "profiles_sample_size": 2,
            "node_sample_size": 3,

            "EOS_CODE": -2,
            "req_min_val": 1,
            "req_max_val": 30,

            "node_min_val": 80,
            "node_max_val": 100,

            "reward": {
                "type": "greedy",
                "greedy": {},
                "fair": {},
                "gini": {}
            }
        }
        self.env = ResourceEnvironmentV3('ResourceV3', ENV_CONFIG)

    def test_constructor(self):
        self.assertEqual(self.env.name, 'ResourceV3')
        self.assertIsNotNone(self.env.rewarder)
        self.assertEqual(self.env.batch_size, 2)
        self.assertEqual(self.env.profiles_sample_size, 2)
        # num_nodes = 3 nodes as in config + 1 EOS node
        num_nodes = 4
        self.assertEqual(self.env.node_sample_size, num_nodes)

    def test_shapes(self):
        self.assertEqual(self.env.total_profiles.shape, (20, 3))

        num_elems = 6  # 3 nodes + 2 features + 1 EOS node
        self.assertEqual(self.env.batch.shape, (2, num_elems, 3))
        self.assertEqual(self.env.bin_net_mask.shape, (2, num_elems))
        self.assertEqual(self.env.resource_net_mask.shape, (2, num_elems))
        self.assertEqual(self.env.mha_used_mask.shape, (2, 1, 1, num_elems))

    def test_initial_masks(self):
        expected_bin_mask = np.array([
            [0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 1, 1],
        ], dtype='float32')

        expected_resource_mask = np.array([
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0, 0],
        ], dtype='float32')

        expected_mha_mask = np.array([
            [[[0, 0, 0, 0, 0, 0]]],
            [[[0, 0, 0, 0, 0, 0]]],
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

        eos_nodes = state[:, 0]
        nodes = state[:, 1:self.env.node_sample_size]
        reqs = state[:, self.env.node_sample_size:]

        self.assertTrue(
            np.all(eos_nodes == self.env.EOS_CODE)
        )

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
        # num_nodes = 3 nodes + 1 EOS node for tracking rejected reqs
        num_nodes = 4
        self.assertEqual(len(self.env.history[0]), num_nodes)
        self.assertEqual(len(self.env.history[1]), num_nodes)

        # Grab the values and shape them into a (batch, node, feature)
        b0_n0 = self.env.history[0][0].get_tensor_rep()
        b0_n1 = self.env.history[0][1].get_tensor_rep()
        b0_n2 = self.env.history[0][2].get_tensor_rep()
        b0_n3 = self.env.history[0][3].get_tensor_rep()

        batch0_nodes = np.array([b0_n0, b0_n1, b0_n2, b0_n3])

        b1_n0 = self.env.history[1][0].get_tensor_rep()
        b1_n1 = self.env.history[1][1].get_tensor_rep()
        b1_n2 = self.env.history[1][2].get_tensor_rep()
        b1_n3 = self.env.history[1][3].get_tensor_rep()

        batch1_nodes = np.array([b1_n0, b1_n1, b1_n2, b1_n3])

        batch_rep = np.array([batch0_nodes, batch1_nodes])

        self.assertEqual(
            nodes.tolist(),
            batch_rep.tolist()
        )

    def test_sample_action(self):
        bin_ids, req_ids, bins_masks = self.env.sample_action()
        batch_size, elems, _ = self.env.batch.shape

        num_nodes = 4 # Including the EOS node
        num_reqs = 2

        # Can't exceed the nodes positions [0,1,2,3]
        self.assertTrue(np.all(bin_ids < num_nodes))
        # Req IDs must be higher that node indexes
        # It can only be either [4, 5]
        self.assertTrue( np.all(req_ids >= num_nodes))

        self.assertEqual(bins_masks.shape, (batch_size, elems))

    def test_step(self):
        # state, bin_net_mask, resource_net_mask, mha_used_mask = self.env.state()

        fake_state = np.array([[[-2.  , -2.  , -2.  ],
                                [ 0.84,  0.9 ,  0.85],
                                [ 0.89,  0.87,  0.89],
                                [ 0.94,  0.86,  0.95],
                                [ 0.29,  0.17,  0.13],
                                [ 0.19,  0.29,  0.22]],

                                [[-2.  , -2.  , -2.  ],
                                [ 0.89,  0.95,  0.97],
                                [ 0.99,  0.95,  0.87],
                                [ 0.89,  0.9 ,  0.82],
                                [ 0.26,  0.23,  0.22],
                                [ 0.1 ,  0.19,  0.25]]], dtype="float32")

        fake_bin_net_mask = np.array([
            [0., 0., 0., 0., 1., 1.],
            [0., 0., 0., 0., 1., 1.]], dtype="float32")

        fake_resource_net_mask = np.array([
            [1., 1., 1., 1., 0., 0.],
            [1., 1., 1., 1., 0., 0.]], dtype="float32")
        
        fake_mha_mask = np.array([
            [[[0., 0., 0., 0., 0., 0.]]],
            [[[0., 0., 0., 0., 0., 0.]]]], dtype="float32")

        # Set the fake states
        self.env.batch = fake_state
        self.env.bin_net_mask = fake_bin_net_mask
        self.env.resource_net_mask = fake_resource_net_mask
        self.env.mha_used_mask = fake_mha_mask

        batch_indices = [0, 1]
        fake_req_ids = [4, 3]
        fake_selected_reqs = fake_state[batch_indices, fake_req_ids]

        fake_bin_ids = [0, 2]
        fake_selected_nodes = fake_state[batch_indices, fake_bin_ids]
        
        # Manually compute the next state
        remaining_resources = fake_selected_nodes - fake_selected_reqs
        fake_state[batch_indices, fake_bin_ids] = remaining_resources
        fake_state[batch_indices, 0] = self.env.EOS_BIN

        next_state, rewards, isDone, info = self.env.step(
            fake_bin_ids, fake_req_ids, fake_bin_net_mask
        )
        self.assertEqual(
            fake_state.tolist(),
            next_state.tolist()
        )

        
        # Update the resource_mask
        fake_resource_net_mask[batch_indices, fake_req_ids] = 1

        next_resource_mask = info['resource_net_mask']

        self.assertEqual(
            fake_resource_net_mask.tolist(),
            next_resource_mask.tolist()
        )
        
        # Didn't fill completely the nodes so they should stay unmasked
        next_bin_mask = info['bin_net_mask']
        self.assertEqual(
            fake_bin_net_mask.tolist(),
            next_bin_mask.tolist()
        )

        # Update the mha mask
        fake_mha_mask[batch_indices, :, :, fake_req_ids] = 1
        # Didn't fill completely the nodes so they should stay unmasked
        next_mha_mask = info['mha_used_mask']
        self.assertEqual(
            fake_mha_mask.tolist(),
            next_mha_mask.tolist()
        )

        # Don't test values reward here.
        # It will be tested in it's own test suite
        self.assertEqual(rewards.shape, (2, 1))

    def test_build_feasible_mask_ALL_SHOULD_be_unmasked(self):
        fake_state = np.array([[[-2.  , -2.  , -2.  ],
                                [ 0.84,  0.9 ,  0.85],
                                [ 0.89,  0.87,  0.89],
                                [ 0.94,  0.86,  0.95],
                                [ 0.29,  0.17,  0.13],
                                [ 0.19,  0.29,  0.22]],

                                [[-2.  , -2.  , -2.  ],
                                [ 0.89,  0.95,  0.97],
                                [ 0.99,  0.95,  0.87],
                                [ 0.89,  0.9 ,  0.82],
                                [ 0.26,  0.23,  0.22],
                                [ 0.1 ,  0.19,  0.25]]], dtype="float32")

        fake_bin_net_mask = np.array([
            [0., 0., 0., 0., 1., 1.],
            [0., 0., 0., 0., 1., 1.]], dtype="float32")
        
        fake_selected_resources = np.array([
            [ 0.29,  0.17,  0.13],
            [ 0.26,  0.23,  0.22],
        ], dtype='float32')


        expected = np.array([
            [0., 0., 0., 0., 1., 1.],
            [0., 0., 0., 0., 1., 1.]], dtype="float32")
        
        actual = self.env.build_feasible_mask(
            fake_state, fake_selected_resources, fake_bin_net_mask)

        self.assertEqual(
            actual.tolist(),
            expected.tolist()
        )