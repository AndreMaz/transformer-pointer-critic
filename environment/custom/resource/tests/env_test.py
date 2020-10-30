import sys

from numpy.core.defchararray import equal
sys.path.append('.')

import numpy as np
import unittest

# Custom Imports
from environment.custom.resource.env import ResourceEnvironment
from environment.custom.resource.reward import Reward
from environment.custom.resource.penalty import Penalty


class TestResource(unittest.TestCase):

    def setUp(self) -> None:
        ENV_CONFIG = {
            "description": "Environment configs.",

            "load_from_file": False,
            "location": "./environment/custom/knapsack/problem.json",

            "batch_size": 2,
            "num_features": 5,
            "num_resources": 100,
            "num_bins": 10,
            "EOS_CODE": 0,

            "resource_sample_size": 5,
            "bin_sample_size": 5,

            "resource_normalization_factor": 1,
            "task_normalization_factor": 1,

            "num_user_levels": 1,
            "reward_per_level": [10, 20],
            "misplace_reward_penalty": 5,

            "num_task_types": 10,

            "CPU_misplace_penalty": 10,
            "RAM_misplace_penalty": 10,
            "MEM_misplace_penalty": 10,

            "min_resource_CPU": 10,
            "max_resource_CPU": 20,
            "min_resource_RAM": 30,
            "max_resource_RAM": 40,
            "min_resource_MEM": 50,
            "max_resource_MEM": 60,

            "min_bin_CPU": 100,
            "max_bin_CPU": 200,
            "min_bin_RAM": 300,
            "max_bin_RAM": 400,
            "min_bin_MEM": 500,
            "max_bin_MEM": 600,

            "min_bin_range_type": 2,
            "max_bin_range_type": 3
        }
        self.env = ResourceEnvironment('Resource', ENV_CONFIG)

    def test_constructor(self):
        self.assertEqual(self.env.name, 'Resource')
        self.assertIsNotNone(self.env.penalizer)
        self.assertIsNotNone(self.env.rewarder)

        self.assertEqual(len(self.env.tasks), 10)

    
    def test_shapes(self):
        # 10 + 1 for EOS bin
        self.assertEqual(self.env.total_bins.shape, (11, 5))
        self.assertEqual(self.env.total_resources.shape, (100, 5))

        self.assertEqual(self.env.batch.shape, (2, 11, 5))
        self.assertEqual(self.env.bin_net_mask.shape, (2,11))
        self.assertEqual(self.env.resource_net_mask.shape, (2,11))
        self.assertEqual(self.env.mha_used_mask.shape, (2, 1, 1, 11))


    def test_reset(self):
        initial_num = self.env.num_inserted_resources()

        # Take 2 random step
        self.env.step([1, 3], [6, 7])
        after_insertion_num = self.env.num_inserted_resources()

        # Reset env
        self.env.reset()
        after_reset_num = self.env.num_inserted_resources()

        self.assertEqual(initial_num, 0)
        self.assertEqual(after_insertion_num, 2)
        self.assertEqual(after_reset_num, 0)


class TestStepFn(unittest.TestCase):

    def setUp(self) -> None:
        ENV_CONFIG = {
            "description": "Environment configs.",

            "load_from_file": False,
            "location": "./environment/custom/knapsack/problem.json",

            "batch_size": 2,
            "num_features": 5,
            "num_resources": 100,
            "num_bins": 10,
            "EOS_CODE": 0,

            "resource_sample_size": 2,
            "bin_sample_size": 2,

            "resource_normalization_factor": 1,
            "task_normalization_factor": 1,

            "num_user_levels": 1,
            "reward_per_level": [10, 20],
            "misplace_reward_penalty": 5,

            "num_task_types": 10,

            "CPU_misplace_penalty": 10,
            "RAM_misplace_penalty": 10,
            "MEM_misplace_penalty": 10,

            "min_resource_CPU": 10,
            "max_resource_CPU": 20,
            "min_resource_RAM": 30,
            "max_resource_RAM": 40,
            "min_resource_MEM": 50,
            "max_resource_MEM": 60,

            "min_bin_CPU": 100,
            "max_bin_CPU": 200,
            "min_bin_RAM": 300,
            "max_bin_RAM": 400,
            "min_bin_MEM": 500,
            "max_bin_MEM": 600,

            "min_bin_range_type": 2,
            "max_bin_range_type": 3
        }
        self.env = ResourceEnvironment('Resource', ENV_CONFIG)

    def test_step_EOS_node_reward_SHOULD_be_zero(self):
        self.env.batch = np.array([[
                [  0.,   0.,   0.,   0.,   0.],
                [100., 200., 300.,   0.,   2.], # Node task range [0, 2]
                [400., 500., 600.,   0.,   3.],
                [ 10.,  20.,  30.,   1.,   1.], # Resource task 1
                [ 40.,  50.,  60.,   8.,   1.]],
            [
                [  0.,   0.,   0.,   0.,   0.],
                [1000., 2000., 3000.,   2.,   5.],
                [4000., 5000., 6000.,   3.,   6.], # Node task range [3, 6]
                [ 100.,  200.,  300.,   0.,   1.],
                [ 400.,  500.,  600.,   4.,   1.]  # Resource task 4
            ]], dtype='float32')
        
        self.env.rebuild_history()

        bin_ids =       [0 , 0]
        resource_ids =  [3 , 4]

        next_state, rewards, isDone, info = self.env.step(
            bin_ids, 
            resource_ids
        )

        expected_next_state = np.array([[
                [  0.,   0.,   0.,   0.,   0.],
                [100., 200., 300.,   0.,   2.],
                [400., 500., 600.,   0.,   3.],
                [ 10.,  20.,  30.,   1.,   1.], # Resource task 1
                [ 40.,  50.,  60.,   8.,   1.]],
            [
                [  0.,   0.,   0.,   0.,   0.],
                [1000., 2000., 3000.,   2.,   5.],
                [4000., 5000., 6000.,   3.,   6.],
                [ 100.,  200.,  300.,   0.,   1.],
                [ 400.,  500.,  600.,   4.,   1.]  # Resource task 4
            ]], dtype='float32')
        self.assertEqual(next_state.tolist(),expected_next_state.tolist())

        expected_rewards = np.array([
            [0],
            [0]
        ], dtype="float32")

        self.assertEqual(rewards.tolist(), expected_rewards.tolist())

        self.assertFalse(isDone)

    def test_step_premium_user_NO_penalty(self):
        self.env.batch = np.array([[
                [  0.,   0.,   0.,   0.,   0.],
                [100., 200., 300.,   0.,   2.], # Node task range [0, 2]
                [400., 500., 600.,   0.,   3.],
                [ 10.,  20.,  30.,   1.,   1.], # Resource task 1
                [ 40.,  50.,  60.,   8.,   1.]],
            [
                [  0.,   0.,   0.,   0.,   0.],
                [1000., 2000., 3000.,   2.,   5.],
                [4000., 5000., 6000.,   3.,   6.], # Node task range [3, 6]
                [ 100.,  200.,  300.,   0.,   1.],
                [ 400.,  500.,  600.,   4.,   1.]  # Resource task 4
            ]], dtype='float32')
        
        self.env.rebuild_history()

        bin_ids =       [1 , 2]
        resource_ids =  [3 , 4]

        next_state, rewards, isDone, info = self.env.step(
            bin_ids, 
            resource_ids
        )

        expected_next_state = np.array([[
                [  0.,   0.,   0.,   0.,   0.],
                [ 90., 180., 270.,   0.,   2.],
                [400., 500., 600.,   0.,   3.],
                [ 10.,  20.,  30.,   1.,   1.],
                [ 40.,  50.,  60.,   8.,   1.]],
            [
                [  0.,   0.,   0.,   0.,   0.],
                [1000., 2000., 3000.,   2.,   5.],
                [3600., 4500., 5400.,   3.,   6.],
                [ 100.,  200.,  300.,   0.,   1.],
                [ 400.,  500.,  600.,   4.,   1.]
            ]], dtype='float32')
        self.assertEqual(next_state.tolist(),expected_next_state.tolist())

        expected_rewards = np.array([
            [20],
            [20]
        ], dtype="float32")

        self.assertEqual(rewards.tolist(), expected_rewards.tolist())

        self.assertFalse(isDone)

    def test_step_premium_user_WITH_penalty(self):
        self.env.batch = np.array([[
                [  0.,   0.,   0.,   0.,   0.],
                [100., 200., 300.,   0.,   2.], # Node task range [0, 2]
                [400., 500., 600.,   0.,   3.],
                [ 10.,  20.,  30.,   15.,   1.], # Resource task 15
                [ 40.,  50.,  60.,   8.,   1.]],
            [
                [  0.,   0.,   0.,   0.,   0.],
                [1000., 2000., 3000.,   2.,   5.],
                [4000., 5000., 6000.,   3.,   6.], # Node task range [3, 6]
                [ 100.,  200.,  300.,   0.,   1.],
                [ 400.,  500.,  600.,   10.,   1.] # Resource task 10
            ]], dtype='float32')
        
        self.env.rebuild_history()

        bin_ids =       [1 , 2]
        resource_ids =  [3 , 4]

        next_state, rewards, isDone, info = self.env.step(
            bin_ids, 
            resource_ids
        )

        expected_next_state = np.array([[
                [  0.,   0.,   0.,   0.,   0.],
                [ 80., 170., 260.,   0.,   2.],
                [ 400., 500., 600.,   0.,   3.],
                [ 10.,  20.,  30.,   15.,   1.],
                [ 40.,  50.,  60.,   8.,   1.]],
            [
                [  0.,   0.,   0.,   0.,   0.],
                [1000., 2000., 3000.,   2.,   5.],
                [3590., 4490., 5390.,   3.,   6.],
                [ 100.,  200.,  300.,   0.,   1.],
                [ 400.,  500.,  600.,   10.,   1.]
            ]], dtype='float32')
        self.assertEqual(next_state.tolist(),expected_next_state.tolist())

        expected_rewards = np.array([
            [15],
            [15]
        ], dtype="float32")

        self.assertEqual(rewards.tolist(), expected_rewards.tolist())
        
        self.assertFalse(isDone)

    def test_step_free_user_NO_penalty(self):
        self.env.batch = np.array([[
                [  0.,   0.,   0.,   0.,   0.],
                [100., 200., 300.,   0.,   2.], # Node task range [0, 2]
                [400., 500., 600.,   0.,   3.],
                [ 10.,  20.,  30.,   1.,   0.], # Resource task 1
                [ 40.,  50.,  60.,   8.,   1.]],
            [
                [  0.,   0.,   0.,   0.,   0.],
                [1000., 2000., 3000.,   2.,   5.],
                [4000., 5000., 6000.,   3.,   6.], # Node task range [3, 6]
                [ 100.,  200.,  300.,   0.,   1.],
                [ 400.,  500.,  600.,   4.,   0.]  # Resource task 4
            ]], dtype='float32')
        
        self.env.rebuild_history()

        bin_ids =       [1 , 2]
        resource_ids =  [3 , 4]

        next_state, rewards, isDone, info = self.env.step(
            bin_ids, 
            resource_ids
        )

        expected_next_state = np.array([[
                [  0.,   0.,   0.,   0.,   0.],
                [ 90., 180., 270.,   0.,   2.],
                [400., 500., 600.,   0.,   3.],
                [ 10.,  20.,  30.,   1.,   0.],
                [ 40.,  50.,  60.,   8.,   1.]],
            [
                [  0.,   0.,   0.,   0.,   0.],
                [1000., 2000., 3000.,   2.,   5.],
                [3600., 4500., 5400.,   3.,   6.],
                [ 100.,  200.,  300.,   0.,   1.],
                [ 400.,  500.,  600.,   4.,   0.]
            ]], dtype='float32')
        self.assertEqual(next_state.tolist(),expected_next_state.tolist())

        expected_rewards = np.array([
            [10],
            [10]
        ], dtype="float32")

        self.assertEqual(rewards.tolist(), expected_rewards.tolist())

        self.assertFalse(isDone)

    def test_step_free_user_WITH_penalty(self):
        self.env.batch = np.array([[
                [  0.,   0.,   0.,   0.,   0.],
                [100., 200., 300.,   0.,   2.], # Node task range [0, 2]
                [400., 500., 600.,   0.,   3.],
                [ 10.,  20.,  30.,   15.,   0.], # Resource task 15
                [ 40.,  50.,  60.,   8.,   1.]],
            [
                [  0.,   0.,   0.,   0.,   0.],
                [1000., 2000., 3000.,   2.,   5.],
                [4000., 5000., 6000.,   3.,   6.], # Node task range [3, 6]
                [ 100.,  200.,  300.,   0.,   1.],
                [ 400.,  500.,  600.,   10.,   0.] # Resource task 10
            ]], dtype='float32')
        
        self.env.rebuild_history()

        bin_ids =       [1 , 2]
        resource_ids =  [3 , 4]

        next_state, rewards, isDone, info = self.env.step(
            bin_ids, 
            resource_ids
        )

        expected_next_state = np.array([[
                [  0.,   0.,   0.,   0.,   0.],
                [ 80., 170., 260.,   0.,   2.],
                [ 400., 500., 600.,   0.,   3.],
                [ 10.,  20.,  30.,   15.,   0.],
                [ 40.,  50.,  60.,   8.,   1.]],
            [
                [  0.,   0.,   0.,   0.,   0.],
                [1000., 2000., 3000.,   2.,   5.],
                [3590., 4490., 5390.,   3.,   6.],
                [ 100.,  200.,  300.,   0.,   1.],
                [ 400.,  500.,  600.,   10.,   0.]
            ]], dtype='float32')
        self.assertEqual(next_state.tolist(),expected_next_state.tolist())

        expected_rewards = np.array([
            [5],
            [5]
        ], dtype="float32")

        self.assertEqual(rewards.tolist(), expected_rewards.tolist())
        
        self.assertFalse(isDone)
    
    def test_step_SHOULD_be_Done(self):
        self.env.batch = np.array([[
                [  0.,   0.,   0.,   0.,   0.],
                [100., 200., 300.,   0.,   2.], # Node task range [0, 2]
                [400., 500., 600.,   0.,   3.],
                [ 10.,  20.,  30.,   15.,   0.], # Resource task 15
                [ 40.,  50.,  60.,   8.,   1.]],
            [
                [  0.,   0.,   0.,   0.,   0.],
                [1000., 2000., 3000.,   2.,   5.],
                [4000., 5000., 6000.,   3.,   6.], # Node task range [3, 6]
                [ 100.,  200.,  300.,   0.,   1.],
                [ 400.,  500.,  600.,   10.,   0.] # Resource task 10
            ]], dtype='float32')
        
        self.env.rebuild_history()
        
        # Before any insertion
        expected_resource_mask = [
            [1.0, 1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 0.0, 0.0]
        ]
        self.assertEqual(
            self.env.resource_net_mask.tolist(),
            expected_resource_mask
        )


        expected_mha_mask = [
            [[[0.0, 0.0, 0.0, 0.0, 0.0]]],
            [[[0.0, 0.0, 0.0, 0.0, 0.0]]]
        ]
        self.assertEqual(
            self.env.mha_used_mask.tolist(),
            expected_mha_mask
        )


        # First Step
        bin_ids =       [1 , 2]
        resource_ids =  [3 , 4]
        next_state, rewards, isDone, info = self.env.step(
            bin_ids, 
            resource_ids
        )

        expected_resource_mask = [
            [1.0, 1.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0, 0.0, 1.0]
        ]
        self.assertEqual(
            info['resource_net_mask'].tolist(),
            expected_resource_mask
        )

        expected_mha_mask = [
            [[[0.0, 0.0, 0.0, 1.0, 0.0]]],
            [[[0.0, 0.0, 0.0, 0.0, 1.0]]]
        ]
        self.assertEqual(
            info['mha_used_mask'].tolist(),
            expected_mha_mask
        )

        # Second Step
        bin_ids =       [0 , 0]
        resource_ids =  [4 , 3]
        next_state, rewards, isDone, info = self.env.step(
            bin_ids, 
            resource_ids
        )

        self.assertTrue(isDone)

        
        expected_resource_mask = [
            [1.0, 1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 1.0]
        ]
        self.assertEqual(
            info['resource_net_mask'].tolist(),
            expected_resource_mask
        )

        expected_mha_mask = [
            [[[0.0, 0.0, 0.0, 1.0, 1.0]]],
            [[[0.0, 0.0, 0.0, 1.0, 1.0]]]
        ]
        self.assertEqual(
            info['mha_used_mask'].tolist(),
            expected_mha_mask
        )


    def test_build_feasible_mask_SHOULD_mask_1_bin(self):
        state = np.array([[
                [  0.,   0.,   0.,   0.,   0.],
                [100., 200., 300.,   0.,   2.], # Node task range [0, 2]
                [  5.,   5.,   5.,   0.,   3.],
                [ 10.,  20.,  30.,   1.,   1.], # Resource task 15
                [ 40.,  50.,  60.,   8.,   1.]],
            [
                [   0.,    0.,    0.,   0.,   0.],
                [   1.,    2.,    3.,   2.,   5.],
                [4000., 5000., 6000.,   3.,   6.], # Node task range [3, 6]
                [ 100.,  200.,  300.,   0.,   1.],
                [ 400.,  500.,  600.,   8.,   1.] # Resource task 10
            ]], dtype='float32')

        resources = np.array([
            [10.,  20.,  30.,   1.,   1.],
            [400.,  500.,  600.,   8.,   1.]
        ], dtype='float32')    

        bin_net_mask = np.array([
            [0., 0., 0., 1.,  1.],
            [0., 0., 0., 1.,  1.]
        ], dtype='float32')

        actual_mask = self.env.build_feasible_mask(
            state,
            resources,
            bin_net_mask
        )

        expected_mask = [
            [0.0, 0.0, 1.0, 1.0, 1.0],
            [0.0, 1.0, 0.0, 1.0, 1.0],
        ]

        self.assertEqual(
            actual_mask.tolist(),
            expected_mask
        )

    def test_build_feasible_mask_mask_all(self):
        state = np.array([[
                [  0.,   0.,   0.,   0.,   0.],
                [  1.,   2.,   3.,   0.,   2.], # Node task range [0, 2]
                [  5.,   5.,   5.,   0.,   3.],
                [ 10.,  20.,  30.,   1.,   0.], # Resource task 15
                [ 40.,  50.,  60.,   8.,   1.]],
            [
                [   0.,    0.,    0.,   0.,   0.],
                [   1.,    2.,    3.,   2.,   5.],
                [   4.,    5.,    6.,   3.,   6.], # Node task range [3, 6]
                [ 100.,  200.,  300.,   0.,   1.],
                [ 400.,  500.,  600.,   8.,   0.] # Resource task 10
            ]], dtype='float32')

        resources = np.array([
            [10.,  20.,  30.,   1.,   1.],
            [400.,  500.,  600.,   8.,   1.]
        ], dtype='float32')    

        bin_net_mask = np.array([
            [0., 0., 0., 1.,  1.],
            [0., 0., 0., 1.,  1.]
        ], dtype='float32')

        actual_mask = self.env.build_feasible_mask(
            state,
            resources,
            bin_net_mask
        )

        expected_mask = [
            [0.0, 1.0, 1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0, 1.0, 1.0],
        ]

        self.assertEqual(
            actual_mask.tolist(),
            expected_mask
        )

    def test_build_feasible_mask_within_task_range(self):
        state = np.array([[
                [  0.,   0.,   0.,   0.,   0.],
                [ 10.,  20.,  30.,   0.,   2.], # Node task range [0, 2]
                [ 10.,  20.,  30.,   0.,   3.],
                [ 10.,  20.,  30.,   3.,   1.], # Resource task 15
                [ 40.,  50.,  60.,   1.,   0.]],
            [
                [   0.,    0.,    0.,   0.,   0.],
                [ 400.,  500.,  600.,   2.,   5.],
                [ 400.,  500.,  600.,   3.,   6.], # Node task range [3, 6]
                [ 100.,  200.,  300.,   0.,   1.],
                [ 400.,  500.,  600.,   6.,   1.] # Resource task 10
            ]], dtype='float32')

        resources = np.array([
            [ 10.,  20.,  30.,   3.,   1.],
            [ 400.,  500.,  600.,   6.,   1.]
        ], dtype='float32')    

        bin_net_mask = np.array([
            [0., 0., 0., 1.,  1.],
            [0., 0., 0., 1.,  1.]
        ], dtype='float32')

        actual_mask = self.env.build_feasible_mask(
            state,
            resources,
            bin_net_mask
        )

        expected_mask = [
            [0.0, 1.0, 0.0, 1.0, 1.0],
            [0.0, 1.0, 0.0, 1.0, 1.0],
        ]

        self.assertEqual(
            actual_mask.tolist(),
            expected_mask
        )