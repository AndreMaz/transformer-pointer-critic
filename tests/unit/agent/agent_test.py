import sys
sys.path.append('.')
import unittest
import numpy as np

# Custom Imports
from environment.custom.resource_v3.env import ResourceEnvironmentV3
from agents.agent import Agent


import tensorflow as tf
# Disable tf.function decorator to avoid rebuilding the graph
tf.config.run_functions_eagerly(True)

class TestResourceGreedyReward(unittest.TestCase):

    def setUp(self) -> None:
        ENV_CONFIG = {
            "description": "Environment configs.",

            "generate_decoder_input": True,
            "mask_nodes_in_mha": False,
            "generate_request_on_the_fly": True,
            "batch_size": 2,

            "seed_value": 1234,

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

        AGENT_CONFIG = {
            
            "single_actor": True,
            "gamma": 0.99,
            "values_loss_coefficient": 1.0,
            "entropy_coefficient": 0.1,
            "stochastic_action_selection": True,
            "use_mha_mask": True,

            "actor": {
                "use_default_initializer": False,
                "num_layers": 1,
                "dim_model": 128,
                "num_heads": 8,
                "inner_layer_dim": 128,
                "SOS_CODE": -1,
                "encoder_embedding_time_distributed": True,
                "attention_dense_units": 128,
                "dropout_rate": 0.0,
                "logit_clipping_C": 10.0,

                "learning_rate": 0.0001,
                "clipnorm": None
            },

            "critic": {
                "use_default_initializer": False,
                "num_layers": 3,
                "dim_model": 128,
                "num_heads": 8,
                "inner_layer_dim": 128,
                "encoder_embedding_time_distributed": True,
                "last_layer_units": 128,
                "last_layer_activation": "linear",
                "dropout_rate": 0.0,

                "learning_rate": 0.0001,
                "clipnorm": None
            }
        }

        self.env = ResourceEnvironmentV3('ResourceV3', ENV_CONFIG)

        config = self.env.add_stats_to_agent_config(AGENT_CONFIG)

        self.agent = Agent('transformer', config)
    
    def tearDown(self) -> None:
        self.env = None
        self.agent = None

    def test_act(self) -> None:
        current_state,\
        decoder_input,\
        bin_net_mask,\
        mha_used_mask = self.env.reset()

        self.assertEqual(
            current_state.shape, (2, 6, 3)
        )
        self.assertEqual(
            decoder_input.shape, (2, 1, 3)
        )

        bin_ids, \
        bins_mask, \
        bins_probs = self.agent.act(
            current_state,
            decoder_input,
            bin_net_mask,
            mha_used_mask,
            self.env.build_feasible_mask
        )

        self.assertEqual(
            len(bin_ids), 2
        )
        self.assertTrue(
            np.all(bin_ids <= 3)
        )

        self.assertEqual(
            bins_mask.shape, (2, 6)
        )

        self.assertEqual(bins_probs.shape, (2, 6))

    def test_loss_compute(self):
        
        # Probabilities from which the actions will sampled
        probs = []
        
        # Initial vars for the initial episode
        isDone = False
        current_state,\
            dec_input,\
            bin_net_mask,\
            mha_used_mask = self.env.reset()

        training_step = 0

        while not isDone:
            # Select an action
            bin_id, bin_net_mask, bin_probs = self.agent.act(
                current_state,
                dec_input,
                bin_net_mask,
                mha_used_mask,
                self.env.build_feasible_mask
            )

            probs.append(bin_probs)

            next_state, next_dec_input, reward, isDone, info = self.env.step(
                bin_id,
                bin_net_mask
            )
            
            # Store in memory
            self.agent.store(
                current_state.copy(),
                dec_input.copy(), # Resource fed to actor decoder
                bin_net_mask.copy(),
                mha_used_mask.copy(),
                bin_id.numpy().copy(),
                reward.numpy().copy(),
                training_step
            )

            # Update for next iteration
            current_state = next_state
            dec_input = next_dec_input.copy()
            bin_net_mask = info['bin_net_mask']
            mha_used_mask = info['mha_used_mask']

            training_step += 1

        bootstrap_state_value = np.zeros(
            [self.agent.batch_size, 1],dtype="float32")
        
        discounted_rewards = self.agent.compute_discounted_rewards(
            bootstrap_state_value
        )
        
        value_loss, state_values, advantages = self.agent.compute_value_loss(
            discounted_rewards
        )

        bin_loss,\
            decoded_bins,\
            bin_entropy,\
            bin_policy_loss,\
            pointers_probs = self.agent.compute_actor_loss(
            self.agent.bin_actor,
            self.agent.bin_masks,
            self.agent.bins,
            self.agent.actor_decoder_input,
            advantages
        )

        self.assertEqual(
            pointers_probs.shape,
            (4, 6)
        )

        # Probabilities should be the same during act() and while computing the loss
        np.testing.assert_almost_equal(
            pointers_probs, np.concatenate(probs, axis=0), decimal = 5
        )

        self.assertEqual(len(self.agent.states), 2)
        self.assertEqual(len(self.agent.actor_decoder_input), 2)
        self.assertEqual(len(self.agent.bin_masks), 2)
        self.assertEqual(len(self.agent.mha_masks), 2)
        self.assertEqual(len(self.agent.bins), 2)

        self.agent.clear_memory()


        self.assertEqual(len(self.agent.states), 0)
        self.assertEqual(len(self.agent.actor_decoder_input), 0)
        self.assertEqual(len(self.agent.bin_masks), 0)
        self.assertEqual(len(self.agent.mha_masks), 0)
        self.assertEqual(len(self.agent.bins), 0)


class TestResourceReduceNodeReward(unittest.TestCase):

    def setUp(self) -> None:
        ENV_CONFIG = {
            "description": "Environment configs.",

            "generate_decoder_input": True,
            "mask_nodes_in_mha": False,
            "generate_request_on_the_fly": True,
            "batch_size": 2,

            "seed_value": 1234,

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
                "type": "reduced_node_usage",
                "greedy": {},
                "fair": {},
                "gini": {},
                "reduced_node_usage": {
                    "rejection_penalty": -2,
                    "use_new_node_penalty": -1
                }
            }
        }

        AGENT_CONFIG = {
            
            "single_actor": True,
            "gamma": 0.99,
            "values_loss_coefficient": 1.0,
            "entropy_coefficient": 0.1,
            "stochastic_action_selection": True,
            "use_mha_mask": True,

            "actor": {
                "use_default_initializer": False,
                "num_layers": 1,
                "dim_model": 128,
                "num_heads": 8,
                "inner_layer_dim": 128,
                "SOS_CODE": -1,
                "encoder_embedding_time_distributed": True,
                "attention_dense_units": 128,
                "dropout_rate": 0.0,
                "logit_clipping_C": 10.0,

                "learning_rate": 0.0001,
                "clipnorm": None
            },

            "critic": {
                "use_default_initializer": False,
                "num_layers": 3,
                "dim_model": 128,
                "num_heads": 8,
                "inner_layer_dim": 128,
                "encoder_embedding_time_distributed": True,
                "last_layer_units": 128,
                "last_layer_activation": "linear",
                "dropout_rate": 0.0,

                "learning_rate": 0.0001,
                "clipnorm": None
            }
        }

        self.env = ResourceEnvironmentV3('ResourceV3', ENV_CONFIG)

        config = self.env.add_stats_to_agent_config(AGENT_CONFIG)

        self.agent = Agent('transformer', config)

    def tearDown(self) -> None:
        self.env = None
        self.agent = None

    def test_act(self) -> None:
        current_state,\
        decoder_input,\
        bin_net_mask,\
        mha_used_mask = self.env.reset()

        self.assertEqual(
            current_state.shape, (2, 6, 4)
        )
        self.assertEqual(
            decoder_input.shape, (2, 1, 3)
        )

        bin_ids, \
        bins_mask, \
        bins_probs = self.agent.act(
            current_state,
            decoder_input,
            bin_net_mask,
            mha_used_mask,
            self.env.build_feasible_mask
        )

        self.assertEqual(
            len(bin_ids), 2
        )
        self.assertTrue(
            np.all(bin_ids <= 3)
        )

        self.assertEqual(
            bins_mask.shape, (2, 6)
        )

        self.assertEqual(bins_probs.shape, (2, 6))

    def test_loss_compute(self):
        
        # Probabilities from which the actions will sampled
        probs = []
        
        # Initial vars for the initial episode
        isDone = False
        current_state,\
            dec_input,\
            bin_net_mask,\
            mha_used_mask = self.env.reset()

        training_step = 0

        while not isDone:
            # Select an action
            bin_id, bin_net_mask, bin_probs = self.agent.act(
                current_state,
                dec_input,
                bin_net_mask,
                mha_used_mask,
                self.env.build_feasible_mask
            )

            probs.append(bin_probs)

            next_state, next_dec_input, reward, isDone, info = self.env.step(
                bin_id,
                bin_net_mask
            )
            
            # Store in memory
            self.agent.store(
                current_state.copy(),
                dec_input.copy(), # Resource fed to actor decoder
                bin_net_mask.copy(),
                mha_used_mask.copy(),
                bin_id.numpy().copy(),
                reward.numpy().copy(),
                training_step
            )

            # Update for next iteration
            current_state = next_state
            dec_input = next_dec_input.copy()
            bin_net_mask = info['bin_net_mask']
            mha_used_mask = info['mha_used_mask']

            training_step += 1

        bootstrap_state_value = np.zeros(
            [self.agent.batch_size, 1],dtype="float32")
        
        discounted_rewards = self.agent.compute_discounted_rewards(
            bootstrap_state_value
        )
        
        value_loss, state_values, advantages = self.agent.compute_value_loss(
            discounted_rewards
        )

        bin_loss,\
            decoded_bins,\
            bin_entropy,\
            bin_policy_loss,\
            pointers_probs = self.agent.compute_actor_loss(
            self.agent.bin_actor,
            self.agent.bin_masks,
            self.agent.bins,
            self.agent.actor_decoder_input,
            advantages
        )

        self.assertEqual(
            pointers_probs.shape,
            (4, 6)
        )

        # Probabilities should be the same during act() and while computing the loss
        np.testing.assert_almost_equal(
            pointers_probs, np.concatenate(probs, axis=0), decimal = 5
        )

        self.assertEqual(len(self.agent.states), 2)
        self.assertEqual(len(self.agent.actor_decoder_input), 2)
        self.assertEqual(len(self.agent.bin_masks), 2)
        self.assertEqual(len(self.agent.mha_masks), 2)
        self.assertEqual(len(self.agent.bins), 2)

        self.agent.clear_memory()


        self.assertEqual(len(self.agent.states), 0)
        self.assertEqual(len(self.agent.actor_decoder_input), 0)
        self.assertEqual(len(self.agent.bin_masks), 0)
        self.assertEqual(len(self.agent.mha_masks), 0)
        self.assertEqual(len(self.agent.bins), 0)