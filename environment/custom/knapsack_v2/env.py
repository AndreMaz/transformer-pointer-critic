from re import S
import sys
from typing import List

from tensorflow.python.ops.gen_batch_ops import batch
sys.path.append('.')

import json
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from environment.base.base import BaseEnvironment
from environment.custom.knapsack_v2.reward import RewardFactory
from environment.custom.knapsack_v2.misc.utils import compute_remaining_resources, round_half_up

from environment.custom.knapsack_v2.bin import Bin as History
from environment.custom.knapsack_v2.item import Item


class KnapsackEnvironmentV2(BaseEnvironment):
    def __init__(self, name: str, opts: dict):
        super(KnapsackEnvironmentV2, self).__init__(name)
        ###########################################
        ##### PROBLEM CONFIGS FROM JSON FILE ######
        ###########################################

        self.gather_stats: bool = False
        self.generate_items_on_the_fly: bool = opts['generate_items_on_the_fly']
        self.mask_nodes_in_mha: bool = opts['mask_nodes_in_mha']

        self.normalization_factor: int = opts['normalization_factor']
        self.decimal_precision: int = opts['decimal_precision']

        self.batch_size: int = opts['batch_size']
        self.num_features: int = opts['num_features']
        
        self.item_set: int = opts['item_set'] # Set of items
        
        self.item_sample_size: int = opts['item_sample_size'] # Sample used in training
        
        assert self.item_set >= self.item_sample_size, 'Item sample size should be less than total number of resources'

        self.EOS_CODE: int = opts['EOS_CODE']
        self.EOS_BIN = np.full((1, self.num_features), self.EOS_CODE, dtype='float32')
        self.bin_sample_size: int = opts['bin_sample_size'] + 1 # + 1 because of the EOS bin

        self.item_min_value: int = opts['item_min_value']
        self.item_max_value: int = opts['item_max_value']

        self.item_min_weight: int = opts['item_min_weight']
        self.item_max_weight: int = opts['item_max_weight']

        self.bin_min_capacity: int = opts['bin_min_capacity']
        self.bin_max_capacity: int = opts['bin_max_capacity']

        ################################################
        ##### MATERIALIZED VARIABLES FROM CONFIGS ######
        ################################################
        self.decoding_step = self.bin_sample_size

        self.rewarder = RewardFactory(
            opts['reward'],
            self.EOS_BIN
        )

        # Generate req profiles
        self.total_items = self.generate_dataset()

        # Problem batch
        self.batch, self.history = self.generate_batch()

        # Default masks
        # Will be updated during at each step() call
        self.bin_net_mask,\
            self.resource_net_mask,\
            self.mha_used_mask = self.generate_masks()

    def reset(self):
        # Reset decoding step
        self.decoding_step = self.bin_sample_size

        self.batch, self.history = self.generate_batch()

        self.bin_net_mask,\
            self.resource_net_mask,\
            self.mha_used_mask = self.generate_masks()

        return self.state()

    def state(self):
        decoder_input = self.batch[:, self.decoding_step]
        decoder_input = np.expand_dims(decoder_input, axis=1)

        batch = self.batch.copy()

        return batch,\
            decoder_input,\
            self.bin_net_mask.copy(),\
            self.mha_used_mask.copy()

    def step(self, bin_ids: List[int], feasible_bin_mask):
        # Default is not done
        isDone = False

        item_ids = tf.fill(self.batch_size, self.decoding_step)
        batch_size = self.batch.shape[0]
        batch_indices = tf.range(batch_size, dtype='int32')

        copy_batch = self.batch.copy()

        # Grab the selected bins and items
        bins: np.ndarray = self.batch[batch_indices, bin_ids]
        items: np.ndarray = self.batch[batch_indices, item_ids]

        # Compute remaining resources after placing item at bins
        updated_bins = compute_remaining_resources(
            bins, items, self.decimal_precision)

        # Update the batch state
        self.batch[batch_indices, bin_ids] = updated_bins
        # Keep EOS node intact
        self.batch[batch_indices, 0] = self.EOS_BIN

        # Item taken mask it
        self.resource_net_mask[batch_indices, item_ids] = 1

        # Update bin masks
        is_node_full = round_half_up(
            updated_bins[:, 0] - updated_bins[:, 1], self.decimal_precision
        )
        # 0 -> Not full
        # 1 -> Full
        is_full = tf.cast(tf.equal(is_node_full, 0), dtype="float32")
        # Mask full nodes/bins
        self.bin_net_mask[batch_indices, bin_ids] = is_full
        self.bin_net_mask[:, 0] = 0 # EOS is always available

        # Update the MHA masks
        self.mha_used_mask[batch_indices, :, :, item_ids] = 1
        if self.mask_nodes_in_mha:
            self.mha_used_mask[batch_indices, :, :, bin_ids] = tf.reshape(
                is_full, (self.batch_size, 1, 1)
            )
        
        # EOS is always available
        self.mha_used_mask[batch_indices, :, :, 0] = 0

        if np.all(self.resource_net_mask == 1):
            isDone = True

        # Compute rewards
        rewards = self.rewarder.compute_reward(
            self.batch, # Already updated values of nodes, i.e., after insertion
            copy_batch, # Original values of nodes, i.e., before insertion
            self.bin_sample_size,
            bins,
            items,
            feasible_bin_mask,
            bin_ids,
        )

        rewards = tf.reshape(rewards, (batch_size, 1))

        info = {
             'bin_net_mask': self.bin_net_mask.copy(),
             'resource_net_mask': self.resource_net_mask.copy(),
             'mha_used_mask': self.mha_used_mask.copy(),
             # 'num_resource_to_place': self.num_profiles
        }

        if self.gather_stats:
            self.place_items(bin_ids, item_ids, items)

        # Pick next decoder_input
        self.decoding_step += 1
        if self.decoding_step < self.bin_sample_size + self.item_sample_size:
            decoder_input = self.batch[:, self.decoding_step]
            decoder_input = np.expand_dims(decoder_input, axis=1)
        else:
            # We are done. No need to generate decoder input
            decoder_input = np.array([None])

        batch = self.batch.copy()
        return batch, decoder_input, rewards, isDone, info

    def generate_dataset(self):
        weights = tf.random.uniform(
            (self.item_set, 1),
            minval = self.item_min_weight,
            maxval = self.item_max_weight,
            dtype='int32'
        ) / self.normalization_factor

        values = tf.random.uniform(
            (self.item_set, 1),
            minval = self.item_min_value,
            maxval = self.item_max_value,
            dtype='int32'
        ) / self.normalization_factor

        items = tf.concat([weights, values], axis=-1)

        return tf.cast(items, dtype='float32')

    def generate_batch(self):
        history = []

        elem_size = self.bin_sample_size + self.item_sample_size

        batch: np.ndarray = np.zeros(
            (self.batch_size, elem_size, self.num_features),
            dtype="float32"
        )

        # Generate bins
        bin_maximum_capacities = tf.random.uniform(
            (self.batch_size, self.bin_sample_size, 1),
            minval=self.bin_min_capacity,
            maxval=self.bin_max_capacity,
            dtype="int32"
        ) / self.normalization_factor

        bin_current_load = tf.zeros(
            (self.batch_size, self.bin_sample_size, 1), dtype='float32'
        )

        bins = tf.concat(
            [tf.cast(bin_maximum_capacities, dtype="float32"), bin_current_load],
            axis=-1
        )

        batch[:, :self.bin_sample_size, :] = tf.cast(bins, dtype="float32")

        # Replace first position with EOS bin
        batch[:, 0, :] = self.EOS_BIN

        if self.generate_items_on_the_fly:
            raise NotImplementedError('"generate_items_on_the_fly" not implemented')
        else:
            # Sample profiles and add them to batch instances
            for index in range(self.batch_size):
                shuffled_profiles = tf.random.shuffle(self.total_items)
                
                batch[index, self.bin_sample_size:, :] = shuffled_profiles[:self.item_sample_size]

        if self.gather_stats:
            history = self.build_history(batch)

        return batch, history

    def generate_masks(self):
        elem_size = self.bin_sample_size + self.item_sample_size

        # Represents positions marked as "0" where resource Ptr Net can point
        items_net_mask = np.zeros((self.batch_size, elem_size), dtype='float32')
        # Represents positions marked as "0" where bin Ptr Net can point
        bins_net_mask = np.ones(
            (self.batch_size, elem_size), dtype='float32')

        # Default mask for resources
        for batch_id in range(self.batch_size):
            for i in range(self.bin_sample_size):
                items_net_mask[batch_id, i] = 1
        
        # Default mask for bin
        bins_net_mask = bins_net_mask - items_net_mask

        # For Transformer's multi head attention
        mha_used_mask = np.zeros_like(items_net_mask)
        mha_used_mask = mha_used_mask[:, np.newaxis, np.newaxis, :]

        return bins_net_mask, items_net_mask, mha_used_mask

    def sample_action(self):
        batch_indices = tf.range(self.batch.shape[0], dtype='int32')

        resource_ids = tf.fill(self.batch_size, self.decoding_step)   
        
        # Decode the resources
        decoded_resources = self.batch[batch_indices, resource_ids]
        decoded_resources = np.expand_dims(decoded_resources, axis=1)

        bins_mask = self.build_feasible_mask(self.batch,
                                             decoded_resources,
                                             self.bin_net_mask
                                             )

        bins_probs = np.random.uniform(size=self.bin_net_mask.shape)
        bins_probs = tf.nn.softmax(bins_probs - (bins_mask*10e6), axis=-1)

        dist_bin = tfp.distributions.Categorical(probs = bins_probs)

        bin_ids = dist_bin.sample()

        return bin_ids, bins_mask
    
    def add_stats_to_agent_config(self, agent_config: dict):
        agent_config['num_resources'] = self.item_sample_size
        agent_config['num_bins'] = self.bin_sample_size

        agent_config['tensor_size'] = self.bin_sample_size + self.item_sample_size
        
        agent_config['batch_size'] = self.batch_size

        # Init the object
        agent_config["encoder_embedding"] = {}
        agent_config["encoder_embedding"]["common"] = False
        agent_config["encoder_embedding"]["num_bin_features"] = 2
        agent_config["encoder_embedding"]["num_resource_features"] = 2
    
        return agent_config

    def set_testing_mode(self,
            batch_size,
            bin_sample_size,
            item_sample_size,
            bin_min_capacity,
            bin_max_capacity
            ) -> None:
        

        self.gather_stats = True
        self.batch_size = batch_size

        self.bin_min_capacity = bin_min_capacity
        self.bin_max_capacity = bin_max_capacity

        self.bin_sample_size = bin_sample_size + 1 # +1 For EOS bin
        self.item_sample_size = item_sample_size

    def build_history(self, batch):
        history = []

        for batch_id, instance in enumerate(batch):
            nodes = []
            for id, bin in enumerate(instance[:self.bin_sample_size]):
                nodes.append(
                    History(
                        batch_id,
                        id,
                        bin
                    )
                )
            
            history.append(nodes)
        
        return history
    
    def place_items(self, bin_ids: List[int], item_ids: List[int], items: np.ndarray):
        for batch_index, bin_id in enumerate(bin_ids):

            node: History = self.history[batch_index][bin_id]
            
            req_id = item_ids[batch_index]
            item = Item(
                batch_index,
                req_id,
                items[batch_index]
            )

            node.insert_item(item)

    def build_feasible_mask(self, state, items, bin_net_mask):
        
        num_elems = state.shape[1]
        tiled_items = tf.tile(items, [1, num_elems, 1])
        item_weights = tiled_items[:, :, 0]

        bin_maximum_capacities = state[:, :, 0]
        bin_current_load = state[:, :, 1]

        remaining_capacity = bin_maximum_capacities - (bin_current_load + item_weights)
        
        can_fit = tf.less(remaining_capacity, 0)
        # 0 => can fit
        # 1 => can't fit
        can_fit = tf.cast(can_fit, dtype='float32')
        
        feasible_mask = tf.maximum(can_fit, bin_net_mask)
        feasible_mask = feasible_mask.numpy()

        assert np.all(remaining_capacity*(1-feasible_mask) >= 0), 'Masking Scheme Is Wrong!'

        # EOS is always available for pointing
        feasible_mask[:, 0] = 0

        return feasible_mask

    def print_history(self, print_details = False) -> None: # pragma: no cover

        for batch_id in range(self.batch_size):
            print('_________________________________')
            for node in self.history[batch_id]:
                node.print(print_details)
            print('_________________________________')

        return

    def store_dataset(self, location) -> None: # pragma: no cover
        np.savetxt(location, self.total_items)
        
    def load_dataset(self, location): # pragma: no cover
        self.total_items = np.loadtxt(location)

if __name__ == "__main__":
    env_name = 'KnapsackV2'
    
    with open(f"configs/KnapsackV2.json") as json_file:
        params = json.load(json_file)

    env_configs = params['env_config']
    env_configs['batch_size'] = 2

    env = KnapsackEnvironmentV2(env_name, env_configs)

    state, dec_input, bin_net_mask, mha_mask = env.state()
    # env.print_history()

    feasible_net_mask = env.build_feasible_mask(state, dec_input, bin_net_mask)

    bin_ids = [0,1]
    next, decoder_input, rewards, isDone, info = env.step(bin_ids, feasible_net_mask)
