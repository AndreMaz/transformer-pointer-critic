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
from environment.custom.resource_v2.reward import RewardFactory
from environment.custom.resource_v2.utils import compute_remaining_resources
from environment.custom.resource_v2.node import Node as History
from environment.custom.resource_v2.resource import Resource as Request

class ResourceEnvironmentV2(BaseEnvironment):
    def __init__(self, name: str, opts: dict):
        super(ResourceEnvironmentV2, self).__init__(name)
        ###########################################
        ##### PROBLEM CONFIGS FROM JSON FILE ######
        ###########################################

        self.gather_stats: bool = False

        self.batch_size: int = opts['batch_size']
        self.num_features: int = opts['num_features']
        self.num_profiles: int = opts['num_profiles']
        
        self.profiles_sample_size: int = opts['profiles_sample_size']
        
        assert self.num_profiles >= self.profiles_sample_size, 'Resource sample size should be less than total number of resources'

        # self.num_nodes: int = opts['num_nodes']
        self.node_sample_size: int = opts['node_sample_size']
        
        # assert self.num_nodes >= self.node_sample_size, 'Bins sample size should be less than total number of bins'

        self.req_min_val: float = opts['req_min_val']
        self.req_max_val: float = opts['req_max_val']

        self.node_min_val: float = opts['node_min_val']
        self.node_max_val: float = opts['node_max_val']

        ################################################
        ##### MATERIALIZED VARIABLES FROM CONFIGS ######
        ################################################
        self.rewarder = RewardFactory(
            opts['reward']
        )

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
        self.batch, self.history = self.generate_batch()

        self.bin_net_mask,\
            self.resource_net_mask,\
            self.mha_used_mask = self.generate_masks()

        return self.state()

    def state(self):
        return self.batch.copy(),\
            self.bin_net_mask.copy(),\
            self.resource_net_mask.copy(),\
            self.mha_used_mask.copy()

    def step(self, bin_ids: List[int], req_ids: List[int], feasible_bin_mask):
        # Default is not done
        isDone = False

        batch_size = self.batch.shape[0]
        num_elems = self.batch.shape[1]
        batch_indices = tf.range(batch_size, dtype='int32')
        
        # Copy the state before updating the values
        copy_batch = self.batch.copy()

        # Grab the selected nodes and resources
        nodes: np.ndarray = self.batch[batch_indices, bin_ids]
        reqs: np.ndarray = self.batch[batch_indices, req_ids]

        # Compute remaining resources after placing reqs at nodes
        remaining_resources = compute_remaining_resources(nodes, reqs)

        # Update the batch state
        self.batch[batch_indices, bin_ids] = remaining_resources
            
        # Item taken mask it
        self.resource_net_mask[batch_indices, req_ids] = 1
        # Update the MHA masks
        self.mha_used_mask[batch_indices, :, :, req_ids] = 1

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
        )

        rewards = tf.reshape(rewards, (batch_size, 1))

        info = {
             'bin_net_mask': self.bin_net_mask.copy(),
             'resource_net_mask': self.resource_net_mask.copy(),
             'mha_used_mask': self.mha_used_mask.copy(),
             # 'num_resource_to_place': self.num_profiles
        }
        
        if self.gather_stats:
            self.place_reqs(bin_ids, req_ids, reqs)

        return self.batch.copy(), rewards, isDone, info
    
    def generate_dataset(self):
    
        profiles = tf.random.uniform(
            (self.num_profiles, self.num_features),
            minval=self.req_min_val,
            maxval=self.req_max_val,
            dtype='float32'
        )

        return profiles

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
            dtype="float32"
        )

        batch[:, :self.node_sample_size, :] = nodes
        
        # Generate reqs
        # reqs = tf.random.uniform(
        #     (self.batch_size, self.profiles_sample_size, self.num_features),
        #     minval=self.req_min_val,
        #     maxval=self.req_max_val,
        #     dtype="float32"
        # )
        
        # batch[:, self.node_sample_size:, :] = reqs
        

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
        
    def add_stats_to_agent_config(self, agent_config: dict):
        agent_config['num_resources'] = self.profiles_sample_size
        agent_config['num_bins'] = self.node_sample_size

        agent_config['tensor_size'] = self.node_sample_size + self.profiles_sample_size
        
        agent_config['batch_size'] = self.batch_size

        # In this env we don't need positional encoding
        # So this can be any value
        agent_config['vocab_size'] = 0 # self.num_nodes + self.num_profiles
    
        return agent_config

    def set_testing_mode(self,
            batch_size,
            node_sample_size,
            profiles_sample_size,
            ) -> None:
        

        self.gather_stats = True
        self.batch_size = batch_size

        self.node_sample_size = node_sample_size
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
                        bin[0], # CPU
                        bin[1], # RAM
                        bin[2], # MEM
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
                reqs[batch_index][0],
                reqs[batch_index][1],
                reqs[batch_index][2]
            )

            node.insert_req(req)


    def build_feasible_mask(self, state, resources, bin_net_mask):
        # Return as is. At this moment node can be overloaded
        return bin_net_mask

    def num_inserted_resources(self):
        return
    
    def rebuild_history(self) -> None:
       return

    def print_history(self, print_details = False) -> None:

        for batch_id in range(self.batch_size):
            print('_________________________________')
            node: History
            for node in self.history[batch_id]:
                node.print(print_details)
            print('_________________________________')

        return

    def validate_history(self):
        return       
    
    def sample_action(self):
        return

    def get_rejection_stats(self) -> dict:
        return

if __name__ == "__main__":
    env_name = 'ResourceEnvironmentV2'
    
    with open(f"configs/ResourceV2.json") as json_file:
        params = json.load(json_file)

    env_configs = params['env_config']

    env = ResourceEnvironmentV2(env_name, env_configs)

    env.print_history()