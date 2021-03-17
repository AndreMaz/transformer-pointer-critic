from re import L
import sys
from typing import List

from tensorflow.python.ops.gen_array_ops import gather

sys.path.append('.')

import json
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from random import randint, randrange
from environment.base.base import BaseEnvironment
from environment.custom.resource_v3.reward import RewardFactory, ReducedNodeUsage
from environment.custom.resource_v3.misc.utils import compute_remaining_resources, round_half_up
from environment.custom.resource_v3.node import Node as History
from environment.custom.resource_v3.resource import Resource as Request

class ResourceEnvironmentV3(BaseEnvironment):
    def __init__(self, name: str, opts: dict):
        super(ResourceEnvironmentV3, self).__init__(name)
        ###########################################
        ##### PROBLEM CONFIGS FROM JSON FILE ######
        ###########################################

        self.gather_stats: bool = False
        self.generate_request_on_the_fly: bool = opts['generate_request_on_the_fly']
        self.mask_nodes_in_mha: bool = opts['mask_nodes_in_mha']

        self.normalization_factor: int = opts['normalization_factor']
        self.decimal_precision: int = opts['decimal_precision']

        self.batch_size: int = opts['batch_size']
        self.num_features: int = opts['num_features']
        self.num_profiles: int = opts['num_profiles']
        
        self.profiles_sample_size: int = opts['profiles_sample_size']
        
        assert self.num_profiles >= self.profiles_sample_size, 'Resource sample size should be less than total number of resources'

        self.EOS_CODE: int = opts['EOS_CODE']
        self.EOS_BIN = np.full((1, self.num_features), self.EOS_CODE, dtype='float32')
        self.node_sample_size: int = opts['node_sample_size'] + 1 # + 1 because of the EOS bin

        self.req_min_val: int = opts['req_min_val']
        self.req_max_val: int = opts['req_max_val']

        self.node_min_val: int = opts['node_min_val']
        self.node_max_val: int = opts['node_max_val']

        ################################################
        ##### MATERIALIZED VARIABLES FROM CONFIGS ######
        ################################################
        self.decoding_step = self.node_sample_size

        self.rewarder = RewardFactory(
            opts['reward'],
            self.EOS_BIN
        )

        if isinstance(self.rewarder, ReducedNodeUsage):
            self.is_empty = np.zeros((self.batch_size, self.node_sample_size + self.profiles_sample_size, 1), dtype='float32')
            # First position is EOS
            self.is_empty[:, 0, 0] = self.EOS_BIN[0][0]
        else:
            self.is_empty = None

        # Generate req profiles
        self.total_profiles = self.generate_dataset()

        # Problem batch
        self.batch, self.history = self.generate_batch()

        # Default masks
        # Will be updated during at each step() call
        self.bin_net_mask,\
            self.resource_net_mask,\
            self.mha_used_mask = self.generate_masks()

    def reset(self):
        # Reset decoding step
        self.decoding_step = self.node_sample_size

        if isinstance(self.rewarder, ReducedNodeUsage):
            self.is_empty = np.zeros(
                (self.batch_size, self.node_sample_size + self.profiles_sample_size, 1), dtype='float32')
            # First position is EOS
            self.is_empty[:, 0, 0] = self.EOS_BIN[0][0]

        self.batch, self.history = self.generate_batch()

        self.bin_net_mask,\
            self.resource_net_mask,\
            self.mha_used_mask = self.generate_masks()

        # Reset the rewarder
        self.rewarder.reset()

        return self.state()

    def state(self):
        decoder_input = self.batch[:, self.decoding_step]
        decoder_input = np.expand_dims(decoder_input, axis=1)

        batch = self.batch.copy()

        if isinstance(self.rewarder, ReducedNodeUsage):
            batch = self.add_is_empty_dim(batch, self.is_empty)

        return batch,\
            decoder_input,\
            self.bin_net_mask.copy(),\
            self.mha_used_mask.copy()

    def step(self, bin_ids: List[int], feasible_bin_mask):
        # Default is not done
        isDone = False

        req_ids = tf.fill(self.batch_size, self.decoding_step)

        batch_size = self.batch.shape[0]
        num_elems = self.batch.shape[1]
        batch_indices = tf.range(batch_size, dtype='int32')
        
        # Copy the state before updating the values
        copy_batch = self.batch.copy()

        # Grab the selected nodes and resources
        nodes: np.ndarray = self.batch[batch_indices, bin_ids]
        reqs: np.ndarray = self.batch[batch_indices, req_ids]

        # Compute remaining resources after placing reqs at nodes
        remaining_resources = compute_remaining_resources(
            nodes, reqs, self.decimal_precision)

        # Update the batch state
        self.batch[batch_indices, bin_ids] = remaining_resources
        # Keep EOS node intact
        self.batch[batch_indices, 0] = self.EOS_BIN
            
        # Item taken mask it
        self.resource_net_mask[batch_indices, req_ids] = 1
        
        # Update node masks
        dominant_resource = tf.reduce_min(remaining_resources, axis=-1)
        is_full = tf.cast(tf.equal(dominant_resource, 0), dtype='float32')
        # Mask full nodes/bins
        self.bin_net_mask[batch_indices, bin_ids] = is_full
        self.bin_net_mask[:, 0] = 0 # EOS is always available

        # Update the MHA masks
        self.mha_used_mask[batch_indices, :, :, req_ids] = 1
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
            self.node_sample_size,
            nodes,
            reqs,
            feasible_bin_mask,
            bin_ids,
            self.is_empty
        )

        rewards = tf.reshape(rewards, (batch_size, 1))
        #else:
        #    rewards = tf.zeros((batch_size, 1), dtype='float32')

        info = {
             'bin_net_mask': self.bin_net_mask.copy(),
             'resource_net_mask': self.resource_net_mask.copy(),
             'mha_used_mask': self.mha_used_mask.copy(),
             # 'num_resource_to_place': self.num_profiles
        }
        
        if self.gather_stats:
            self.place_reqs(bin_ids, req_ids, reqs)
        
        # Pick next decoder_input
        self.decoding_step += 1
        if self.decoding_step < self.node_sample_size + self.profiles_sample_size:
            decoder_input = self.batch[:, self.decoding_step]
            decoder_input = np.expand_dims(decoder_input, axis=1)
        else:
            # We are done. No need to generate decoder input
            decoder_input = np.array([None])

        batch = self.batch.copy()
        if isinstance(self.rewarder, ReducedNodeUsage):
            batch = self.add_is_empty_dim(batch, self.is_empty)

        return batch, decoder_input, rewards, isDone, info
    
    def generate_dataset(self):
    
        profiles = tf.random.uniform(
            (self.num_profiles, self.num_features),
            minval=self.req_min_val,
            maxval=self.req_max_val,
            dtype='int32'
        ) / self.normalization_factor

        return tf.cast(profiles, dtype="float32")

    def generate_batch(self):
        history = []

        elem_size = self.node_sample_size + self.profiles_sample_size

        batch: np.ndarray = np.zeros(
            (self.batch_size, elem_size, self.num_features),
            dtype="float32"
        )

        # Generate nodes states
        nodes = tf.random.uniform(
            (self.batch_size, self.node_sample_size, self.num_features),
            minval=self.node_min_val,
            maxval=self.node_max_val,
            dtype="int32"
        ) / self.normalization_factor

        batch[:, :self.node_sample_size, :] = tf.cast(nodes, dtype="float32")
        
        # Replace first position with EOS node
        batch[:, 0, :] = self.EOS_BIN

        if self.generate_request_on_the_fly:
            # Generate reqs
            reqs = tf.random.uniform(
                (self.batch_size, self.profiles_sample_size, self.num_features),
                minval=self.req_min_val,
                maxval=self.req_max_val,
                dtype="int32"
            ) / self.normalization_factor
            
            batch[:, self.node_sample_size:, :] = tf.cast(reqs, dtype="float32")
        else:
            # Sample profiles and add them to batch instances
            for index in range(self.batch_size):
                shuffled_profiles = tf.random.shuffle(self.total_profiles)
                
                batch[index, self.node_sample_size:, :] = shuffled_profiles[:self.profiles_sample_size]

        # Create node instances that will gather stats
        if self.gather_stats:
            history = self.build_history(batch)

        return batch, history

    def generate_masks(self):
        elem_size = self.node_sample_size + self.profiles_sample_size

        # Represents positions marked as "0" where resource Ptr Net can point
        profiles_net_mask = np.zeros((self.batch_size, elem_size), dtype='float32')
        # Represents positions marked as "0" where bin Ptr Net can point
        nodes_net_mask = np.ones(
            (self.batch_size, elem_size), dtype='float32')

                # Default mask for resources
        for batch_id in range(self.batch_size):
            for i in range(self.node_sample_size):
                profiles_net_mask[batch_id, i] = 1
        
        # Default mask for bin
        nodes_net_mask = nodes_net_mask - profiles_net_mask

        # For Transformer's multi head attention
        mha_used_mask = np.zeros_like(profiles_net_mask)
        mha_used_mask = mha_used_mask[:, np.newaxis, np.newaxis, :]

        return nodes_net_mask, profiles_net_mask, mha_used_mask
    
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
        agent_config['num_resources'] = self.profiles_sample_size
        agent_config['num_bins'] = self.node_sample_size

        agent_config['tensor_size'] = self.node_sample_size + self.profiles_sample_size
        
        agent_config['batch_size'] = self.batch_size

        # Init the object
        agent_config["encoder_embedding"] = {}
        if isinstance(self.rewarder, ReducedNodeUsage):
            agent_config["encoder_embedding"]["common"] = False
            agent_config["encoder_embedding"]["num_bin_features"] = 4
            agent_config["encoder_embedding"]["num_resource_features"] = 3
        else:
            agent_config["encoder_embedding"]["common"] = True
            # If using the same embedding layer these vars are unused
            agent_config["encoder_embedding"]["num_bin_features"] = None
            agent_config["encoder_embedding"]["num_resource_features"] = None

        return agent_config

    def set_testing_mode(self,
            batch_size,
            node_sample_size,
            profiles_sample_size,
            node_min_val,
            node_max_val
            ) -> None:
        

        self.gather_stats = True
        self.batch_size = batch_size

        self.node_min_val = node_min_val
        self.node_max_val = node_max_val

        self.node_sample_size = node_sample_size + 1 # +1 For EOS node
        self.profiles_sample_size = profiles_sample_size

    def build_history(self, batch):
        history = []

        for batch_id, instance in enumerate(batch):
            nodes = []
            for id, bin in enumerate(instance[:self.node_sample_size]):
                nodes.append(
                    History(
                        batch_id,
                        id,
                        bin
                    )
                )
            
            history.append(nodes)
        
        return history

    def place_reqs(self, bin_ids: List[int], req_ids: List[int], reqs: np.ndarray):
        for batch_index, bin_id in enumerate(bin_ids):

            node: History = self.history[batch_index][bin_id]
            
            req_id = req_ids[batch_index]
            req = Request(
                batch_index,
                req_id,
                reqs[batch_index]
            )

            node.insert_req(req)


    def build_feasible_mask(self, state, resources, bin_net_mask):
        
        if isinstance(self.rewarder, ReducedNodeUsage):
            state = self.remove_is_empty_dim(state)

        batch = state.shape[0]
        num_elems = state.shape[1]
        # Add batch dim to resources
        # resource_demands = np.reshape(resources, (batch, 1, self.num_features))
        # Tile to match the num elems
        resource_demands = tf.tile(resources, [1, num_elems, 1])

        # Compute remaining resources after placement
        # remaining_resources = state - resource_demands
        remaining_resources = compute_remaining_resources(
            state, resource_demands, self.decimal_precision
            )

        dominant_resource = tf.reduce_min(remaining_resources, axis=-1)
        
        # Ensure that it's greater that 0
        # i.e., that node is not overloaded
        after_place = tf.less(dominant_resource, 0)
        after_place = tf.cast(after_place, dtype='float32')

        # Can't point to resources positions
        feasible_mask = tf.maximum(after_place, bin_net_mask)
        feasible_mask = feasible_mask.numpy()
        
        assert np.all(dominant_resource*(1-feasible_mask) >= 0), 'Masking Scheme Is Wrong!'

        # EOS is always available for pointing
        feasible_mask[:, 0] = 0

        # Return as is. At this moment node can be overloaded
        return feasible_mask

    def add_is_empty_dim(self, batch, is_empty):
        batch = np.concatenate([batch, is_empty], axis=-1)
        return round_half_up(batch, 2)
    
    def remove_is_empty_dim(self, batch):
        batch = batch[:, :, :self.num_features]
        return round_half_up(batch, 2)

    def print_history(self, print_details = False) -> None: # pragma: no cover

        for batch_id in range(self.batch_size):
            print('_________________________________')
            node: History
            for node in self.history[batch_id]:
                node.print(print_details)
            print('_________________________________')

        return

    def store_dataset(self, location) -> None:
        np.savetxt(location, self.total_profiles)
        
    def load_dataset(self, location):
        self.total_profiles = np.loadtxt(location)

if __name__ == "__main__":
    env_name = 'ResourceEnvironmentV3'
    
    with open(f"configs/ResourceV3.json") as json_file:
        params = json.load(json_file)

    env_configs = params['env_config']
    env_configs['batch_size'] = 2

    env = ResourceEnvironmentV3(env_name, env_configs)

    state, dec, bin_net_mask, mha_mask = env.state()
    # env.print_history()

    feasible_net_mask = env.build_feasible_mask(state, dec, bin_net_mask)

    bin_ids = [0,1]
    resource_ids = None
    next, decoder_input, rewards, isDone, info = env.step(bin_ids, feasible_net_mask)

    next, decoder_input, rewards, isDone, info = env.step(bin_ids, feasible_net_mask)
    
    env.reset()

    next, decoder_input, rewards, isDone, info = env.step(bin_ids, feasible_net_mask)

    next, decoder_input, rewards, isDone, info = env.step(bin_ids, feasible_net_mask)


    a = 1