import sys
sys.path.append('.')
import unittest
import numpy as np


# Custom Imports
from environment.custom.resource_v3.env import ResourceEnvironmentV3
from agents.agent import Agent


class TestResource(unittest.TestCase):

    def setUp(self) -> None:
        ENV_CONFIG = {
            "description": "Environment configs.",

            "mask_nodes_in_mha": False,
            "generate_request_on_the_fly": True,
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

        AGENT_CONFIG = {

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
                "positional_encoding": False,
                "SOS_CODE": -1,
                "encoder_embedding_time_distributed": True,
                "attention_dense_units": 128,
                "dropout_rate": 0.0,
                "logit_clipping_C": 10.0,

                "learning_rate": 0.0001
            },

            "critic": {
                "use_default_initializer": False,
                "num_layers": 3,
                "dim_model": 128,
                "num_heads": 8,
                "positional_encoding": False,
                "inner_layer_dim": 128,
                "encoder_embedding_time_distributed": True,
                "last_layer_units": 128,
                "last_layer_activation": "linear",
                "dropout_rate": 0.0,

                "learning_rate": 0.0001
            }
        }

        self.env = ResourceEnvironmentV3('ResourceV3', ENV_CONFIG)

        config = self.env.add_stats_to_agent_config(AGENT_CONFIG)

        self.agent = Agent('transformer', config)


    def test_act(self) -> None:
        current_state,\
        bin_net_mask,\
        resource_net_mask,\
        mha_used_mask = self.env.reset()

        decoder_input = self.agent.generate_decoder_input(current_state)

        self.assertEqual(
            decoder_input.shape, (2, 1, 3)
        )

        bin_ids, \
        resource_ids, \
        decoded_resources, \
        bins_mask, \
        resources_probs, \
        bins_probs = self.agent.act(
            current_state,
            decoder_input,
            bin_net_mask,
            resource_net_mask,
            mha_used_mask,
            self.env.build_feasible_mask
        )

        self.assertEqual(
            len(bin_ids), 2
        )
        self.assertTrue(
            np.all(bin_ids <= 3)
        )

        self.assertEqual(len(resource_ids), 2)
        self.assertTrue(
            np.all(resource_ids > 3) and np.all(resource_ids <= 5)
        )

        self.assertEqual(
            decoded_resources.shape, (2, 1, 3)
        )

        self.assertEqual(
            bins_mask.shape, (2, 6)
        )

        self.assertEqual(resources_probs.shape, (2, 6))
        self.assertEqual(bins_probs.shape, (2, 6))