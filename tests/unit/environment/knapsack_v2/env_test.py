import sys
sys.path.append('.')
from environment.custom.knapsack_v2.misc.utils import compute_remaining_resources

import unittest
import numpy as np

# Custom Imports
from environment.custom.knapsack_v2.env import KnapsackEnvironmentV2

class TestKnapsackEnvironmentV2(unittest.TestCase):
    def setUp(self) -> None:
        ENV_CONFIG = {
            "description": "Environment configs.",

            "batch_size": 2,
            "mask_nodes_in_mha": False,
            "generate_items_on_the_fly": False,

            "normalization_factor": 100,
            "decimal_precision": 2,

            "num_features": 2,
            "item_set": 20,

            "item_sample_size": 2,
            "bin_sample_size": 3,

            "EOS_CODE": -2,

            "item_min_value": 70,
            "item_max_value": 80,

            "item_min_weight": 1,
            "item_max_weight": 30,

            "bin_min_capacity": 90,
            "bin_max_capacity": 100,

            "reward": {
                "type": "greedy",
                "greedy": {}
            }
        }
        self.env = KnapsackEnvironmentV2('KnapsackV2', ENV_CONFIG)

    def test_constructor(self):
        self.assertEqual(self.env.name, 'KnapsackV2')
        self.assertIsNotNone(self.env.rewarder)
        self.assertEqual(self.env.batch_size, 2)
        self.assertEqual(self.env.item_sample_size, 2)
        # num_bins = 3 bins as in config + 1 EOS bin
        num_bins = 4
        self.assertEqual(self.env.bin_sample_size, num_bins)

    def test_shapes(self):
        self.assertEqual(self.env.total_items.shape, (20, 2))

        num_elems = 6  # 3 nodes + 2 features + 1 EOS node
        self.assertEqual(self.env.batch.shape, (2, num_elems, 2))
        self.assertEqual(self.env.bin_net_mask.shape, (2, num_elems))
        self.assertEqual(self.env.resource_net_mask.shape, (2, num_elems))
        self.assertEqual(self.env.mha_used_mask.shape, (2, 1, 1, num_elems))

    def test_reset(self):
        state, decoder_input, bin_net_mask, mha_used_mask = self.env.reset()

        self.assertEqual(state.shape, (2, 6, 2))
        self.assertEqual(decoder_input.shape, (2, 1, 2))
        self.assertEqual(bin_net_mask.shape, (2, 6))
        self.assertEqual(mha_used_mask.shape, (2, 1, 1, 6))
    
    def test_initial_masks(self):
        expected_bin_mask = np.array([
            [0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 1, 1],
        ], dtype='float32')

        expected_mha_mask = np.array([
            [[[0, 0, 0, 0, 0, 0]]],
            [[[0, 0, 0, 0, 0, 0]]],
        ], dtype='float32')

        state, decoder_input, bin_net_mask, mha_mask = self.env.state()

        self.assertEqual(
            bin_net_mask.tolist(),
            expected_bin_mask.tolist()
        )

        self.assertEqual(
            mha_mask.tolist(),
            expected_mha_mask.tolist()
        )
    
    def test_batch_values(self):

        state, _, _, _ = self.env.state()

        eos_nodes = state[:, 0]
        nodes = state[:, 1:self.env.bin_sample_size]
        items = state[:, self.env.bin_sample_size:]

        self.assertTrue(
            np.all(eos_nodes == self.env.EOS_CODE)
        )

        # Test Bin Capacity
        self.assertTrue(
            np.all(nodes[:, :, 0] >= 0.9)
        )

        # Test Bin Current Load
        self.assertTrue(
            np.all(nodes[:, :, 1] == 0.0)
        )

        # Test Item Weight
        self.assertTrue(
            np.all(items[:, :, 0] > 0.0) and np.all(items[:, :, 0] <= 0.3)
        )

        # Test Item Value
        self.assertTrue(
            np.all(items[:, :, 1] >= 0.7) and np.all(items[:, :, 1] <= 0.8)
        )
    
    def test_testing_mode(self):
        # The values in batch should match the values in the class instantes
        self.env.set_testing_mode(
            batch_size=2,
            bin_sample_size=3,
            item_sample_size=2,
            bin_min_capacity=  80,
            bin_max_capacity= 100,
        )
        self.env.reset()

        state, _, _, _ = self.env.state()

        nodes = state[:, :self.env.bin_sample_size]
        reqs = state[:, self.env.bin_sample_size:]

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

        batch0_bins = np.array([b0_n0, b0_n1, b0_n2, b0_n3])

        b1_n0 = self.env.history[1][0].get_tensor_rep()
        b1_n1 = self.env.history[1][1].get_tensor_rep()
        b1_n2 = self.env.history[1][2].get_tensor_rep()
        b1_n3 = self.env.history[1][3].get_tensor_rep()

        batch1_bins = np.array([b1_n0, b1_n1, b1_n2, b1_n3])

        batch_rep = np.array([batch0_bins, batch1_bins])

        self.assertEqual(
            nodes.tolist(),
            batch_rep.tolist()
        )
    
    def test_sample_action(self):
        bin_ids, bins_masks = self.env.sample_action()
        batch_size, elems, _ = self.env.batch.shape

        num_nodes = 4 # Including the EOS node

        # Can't exceed the nodes positions [0,1,2,3]
        self.assertTrue(np.all(bin_ids < num_nodes))

        self.assertEqual(bins_masks.shape, (batch_size, elems))

    def test_step(self):
        self.env.set_testing_mode(
            batch_size=2,
            bin_sample_size=3,
            item_sample_size=2,
            bin_min_capacity=  80,
            bin_max_capacity= 100
        )
        
        fake_state = np.array([[[-2.  , -2. ],
                                [ 0.84,  0.0],
                                [ 0.89,  0.0],
                                [ 0.94,  0.0],
                                [ 0.09,  0.75],
                                [ 0.06,  0.70]],

                                [[-2.  , -2. ],
                                [ 0.89,  0.0],
                                [ 0.99,  0.0],
                                [ 0.89,  0.0 ],
                                [ 0.07,  0.73],
                                [ 0.08,  0.72]]], dtype="float32")

        fake_bin_net_mask = np.array([
            [0., 0., 0., 0., 1., 1.],
            [0., 0., 0., 0., 1., 1.]], dtype="float32")

        fake_resource_net_mask = np.array([
            [1., 1., 1., 1., 0., 0.],
            [1., 1., 1., 1., 0., 0.]], dtype="float32")
        
        fake_mha_mask = np.array([
            [[[0., 0., 0., 0., 0., 0.]]],
            [[[0., 0., 0., 0., 0., 0.]]]], dtype="float32")

        # Set the fake states and build the history
        self.env.history = self.env.build_history(fake_state)

        self.env.batch = fake_state
        self.env.bin_net_mask = fake_bin_net_mask
        self.env.resource_net_mask = fake_resource_net_mask
        self.env.mha_used_mask = fake_mha_mask

        batch_indices = [0, 1]
        fake_req_ids = [4, 4]
        fake_selected_reqs = fake_state[batch_indices, fake_req_ids]

        fake_bin_ids = [0, 2]
        fake_selected_nodes = fake_state[batch_indices, fake_bin_ids]
        
        # Manually compute the next state
        updated_bins = compute_remaining_resources(fake_selected_nodes, fake_selected_reqs)
        # remaining_resources = fake_selected_nodes - fake_selected_reqs
        fake_state[batch_indices, fake_bin_ids] = updated_bins
        fake_state[batch_indices, 0] = self.env.EOS_BIN

        next_state, next_decoder_input, rewards, isDone, info = self.env.step(
            fake_bin_ids, fake_bin_net_mask
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

        # # Don't test values reward here.
        # # It will be tested in it's own test suite
        self.assertEqual(rewards.shape, (2, 1))

        # Ensure that history is also working properly
        # Problem instance 0. Req was placed at EOS node (index 0)
        self.assertEqual( len(self.env.history[0][0].item_list), 1 )
        # Others nodes should be empty
        self.assertEqual( len(self.env.history[0][1].item_list), 0 )
        self.assertEqual( len(self.env.history[0][2].item_list), 0 )
        self.assertEqual( len(self.env.history[0][3].item_list), 0 )

        # Check the actual request values
        # Should be equal to the fake selected one
        instance_0_actual_req = self.env.history[0][0].item_list[0]
        self.assertEqual(
                instance_0_actual_req.weight.tolist() +
                instance_0_actual_req.value.tolist()
            ,
            fake_selected_reqs[0].tolist()
        )

        # Problem instance 1. Req was placed at EOS node (index 2)
        self.assertEqual( len(self.env.history[1][0].item_list), 0 )
        # Others nodes should be empty
        self.assertEqual( len(self.env.history[1][1].item_list), 0 )
        self.assertEqual( len(self.env.history[1][2].item_list), 1 )
        self.assertEqual( len(self.env.history[1][3].item_list), 0 )

        instance_1_actual_req = self.env.history[1][2].item_list[0]
        self.assertEqual(
                instance_1_actual_req.weight.tolist() +
                instance_1_actual_req.value.tolist()
            ,
            fake_selected_reqs[1].tolist()
        )
    
    def test_is_done(self):
        self.env.reset()

        bin_ids, bins_mask = self.env.sample_action()
        _, _, _, is_done, _ = self.env.step(bin_ids, bins_mask)

        self.assertFalse(is_done)

        bin_ids, bins_mask = self.env.sample_action()
        _, _, _, is_done, _ = self.env.step(bin_ids, bins_mask)

        self.assertTrue(is_done)

    def test_build_feasible_mask_ALL_SHOULD_be_unmasked(self):
        fake_state = np.array([[[-2.  , -2.],
                                [ 0.8,  0.0],
                                [ 0.8,  0.0],
                                [ 0.9,  0.0],
                                [ 0.1,  0.99],
                                [ 0.3,  0.3]],

                                [[-2.  , -2.],
                                [ 0.8,  0.0],
                                [ 0.9,  0.0],
                                [ 0.8,  0.0],
                                [ 0.2,  0.99],
                                [ 0.5 , 0.5]]], dtype="float32")

        fake_bin_net_mask = np.array([
            [0., 0., 0., 0., 1., 1.],
            [0., 0., 0., 0., 1., 1.]], dtype="float32")
        
        fake_selected_resources = np.array([
            [ 0.1,  0.99],
            [ 0.2,  0.99],
        ], dtype='float32')
        fake_selected_resources = np.expand_dims(fake_selected_resources, axis=1)

        expected = np.array([
            [0., 0., 0., 0., 1., 1.],
            [0., 0., 0., 0., 1., 1.]], dtype="float32")
        
        actual = self.env.build_feasible_mask(
            fake_state, fake_selected_resources, fake_bin_net_mask)

        self.assertEqual(
            actual.tolist(),
            expected.tolist()
        )
    
    def test_build_feasible_mask_ALL_SHOULD_be_masked(self):
        fake_state = np.array([[[-2.  , -2.],
                                [ 0.9,  0.8],
                                [ 0.8,  0.6],
                                [ 0.8,  0.8],
                                [ 0.1,  0.1],
                                [ 0.3,  0.1]],

                                [[-2.  , -2.],
                                [ 0.6,  0.2],
                                [ 0.7,  0.3],
                                [ 0.4,  0.0],
                                [ 0.2,  0.2],
                                [ 0.5 , 0.2]]], dtype="float32")

        fake_bin_net_mask = np.array([
            [0., 0., 0., 0., 1., 1.],
            [0., 0., 0., 0., 1., 1.]], dtype="float32")
        
        fake_selected_resources = np.array([
            [ 0.3,  0.1],
            [ 0.5,  0.2],
        ], dtype='float32')
        fake_selected_resources = np.expand_dims(fake_selected_resources, axis=1)

        expected = np.array([
            [0., 1., 1., 1., 1., 1.],
            [0., 1., 1., 1., 1., 1.]], dtype="float32")
        
        actual = self.env.build_feasible_mask(
            fake_state, fake_selected_resources, fake_bin_net_mask)

        self.assertEqual(
            actual.tolist(),
            expected.tolist()
        )
    
    def test_build_feasible_mask_2nd_and_3rd_SHOULD_be_unmasked(self):
        fake_state = np.array([[[-2.  , -2.],
                                [ 0.9,  0.8],
                                [ 0.4,  0.1],
                                [ 0.3,  0.0],
                                [ 0.1,  0.1],
                                [ 0.3,  0.3]],

                                [[-2.  , -2.],
                                [ 0.3,  0.2],
                                [ 0.5,  0.0],
                                [ 0.6,  0.0],
                                [ 0.2,  0.2],
                                [ 0.5 , 0.5]]], dtype="float32")

        fake_bin_net_mask = np.array([
            [0., 0., 0., 0., 1., 1.],
            [0., 0., 0., 0., 1., 1.]], dtype="float32")
        
        fake_selected_resources = np.array([
            [ 0.3,  0.3],
            [ 0.5,  0.5],
        ], dtype='float32')
        fake_selected_resources = np.expand_dims(fake_selected_resources, axis=1)

        expected = np.array([
            [0., 1., 0., 0., 1., 1.],
            [0., 1., 0., 0., 1., 1.]], dtype="float32")
        
        actual = self.env.build_feasible_mask(
            fake_state, fake_selected_resources, fake_bin_net_mask)

        self.assertEqual(
            actual.tolist(),
            expected.tolist()
        )
    
    def test_build_feasible_mask_SHOULD_mask_2_nodes(self):
        fake_state = np.array([[[-2.  , -2.],
                                [ 0.3,  0.2],
                                [ 0.4,  0.0],
                                [ 0.7,  0.5],
                                [ 0.1,  0.1],
                                [ 0.3,  0.3]],

                                [[-2.  , -2.],
                                [ 0.7,  0.1],
                                [ 0.3,  0.3],
                                [ 0.1,  0.1],
                                [ 0.2,  0.2],
                                [ 0.5 , 0.5]]], dtype="float32")

        fake_bin_net_mask = np.array([
            [0., 0., 0., 0., 1., 1.],
            [0., 0., 0., 0., 1., 1.]], dtype="float32")
        
        fake_selected_resources = np.array([
            [ 0.3,  0.3],
            [ 0.5,  0.5],
        ], dtype='float32')
        fake_selected_resources = np.expand_dims(fake_selected_resources, axis=1)

        expected = np.array([
            [0., 1., 0., 1., 1., 1.],
            [0., 0., 1., 1., 1., 1.]], dtype="float32")
        
        actual = self.env.build_feasible_mask(
            fake_state, fake_selected_resources, fake_bin_net_mask)

        self.assertEqual(
            actual.tolist(),
            expected.tolist()
        )
    
    def test_build_feasible_mask_SHOULD_mask_2_nodes_single_resource_overloaded(self):
        fake_state = np.array([[[-2.  , -2.],
                                [ 0.9,  0.7],
                                [ 0.7,  0.6],
                                [ 0.3,  0.0],
                                [ 0.1,  0.1],
                                [ 0.3,  0.3]],

                                [[-2.  , -2.],
                                [ 0.7,  0.2],
                                [ 0.3,  0.0],
                                [ 0.9,  0.7],
                                [ 0.2,  0.2],
                                [ 0.5 , 0.5]]], dtype="float32")

        fake_bin_net_mask = np.array([
            [0., 0., 0., 0., 1., 1.],
            [0., 0., 0., 0., 1., 1.]], dtype="float32")
        
        fake_selected_resources = np.array([
            [ 0.3,  0.3],
            [ 0.5,  0.5],
        ], dtype='float32')
        fake_selected_resources = np.expand_dims(fake_selected_resources, axis=1)

        expected = np.array([
            [0., 1., 1., 0., 1., 1.],
            [0., 0., 1., 1., 1., 1.]], dtype="float32")
        
        actual = self.env.build_feasible_mask(
            fake_state, fake_selected_resources, fake_bin_net_mask)

        self.assertEqual(
            actual.tolist(),
            expected.tolist()
        )
    
    def test_add_stats_to_agent_config(self):
        actual = {}

        actual = self.env.add_stats_to_agent_config(actual)

        expected = {
                "common": False,
                "num_bin_features": 2,
                "num_resource_features": 2
        }

        self.assertEqual(actual['encoder_embedding'], expected)
