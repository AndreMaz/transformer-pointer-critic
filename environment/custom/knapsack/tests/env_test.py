import sys
sys.path.append('.')

import unittest
import numpy as np

### Custom Imports
from environment.custom.knapsack.env import Knapsack
from environment.custom.knapsack.backpack import Backpack, EOS_BACKPACK, NORMAL_BACKPACK
from environment.custom.knapsack.item import Item

class TestKnapsackEnv(unittest.TestCase):

    def test_constructor(self):
        env = Knapsack('knapsack', {
            "batch_size": 2,
            "num_items": 2,
            "num_backpacks": 1,

            "min_item_value": 3,
            "max_item_value": 4,
            "min_item_weight": 5,
            "max_item_weight": 6,

            "min_backpack_capacity": 7,
            "max_backpack_capacity": 8
        })

        self.assertEqual(env.batch_size, 2)
        self.assertEqual(env.num_items, 2)
        self.assertEqual(env.num_backpacks, 1)

        self.assertEqual(env.min_item_value, 3)
        self.assertEqual(env.max_item_value, 4)
        self.assertEqual(env.min_item_weight, 5)
        self.assertEqual(env.max_item_weight, 6)

        self.assertEqual(env.min_backpack_capacity, 7)
        self.assertEqual(env.max_backpack_capacity, 8)

        self.assertEqual(env.tensor_size, 4) # 2 Items + 1 Backpack + 1 EOS Backpack

        self.assertEqual(len(env.problem_list), 2)

    def test_generate_fn(self):
        env = Knapsack('knapsack', {
            "batch_size": 2,
            "num_items": 2,
            "num_backpacks": 2,

            "min_item_value": 3,
            "max_item_value": 3,

            "min_item_weight": 5,
            "max_item_weight": 5,

            "min_backpack_capacity": 10,
            "max_backpack_capacity": 10
        })

        prob_list = env.generate()

        self.assertEqual(len(prob_list), 2)

        sub_prob_1 = prob_list[0]
        self.assertEqual(len(sub_prob_1['backpacks']), 3) # 2 normal + 1 EOS backpack
        self.assertEqual(len(sub_prob_1['items']), 2) # 2 normal + 1 EOS backpack

    def test_state_fn(self):
        env = Knapsack('knapsack', {
            "batch_size": 2,
            "num_items": 2,
            "num_backpacks": 2,

            "min_item_value": 3,
            "max_item_value": 3,

            "min_item_weight": 5,
            "max_item_weight": 5,

            "min_backpack_capacity": 10,
            "max_backpack_capacity": 10
        })

        actual_state, _, _ = env.state()
        
        # Should be:
        # 2 -> batch size
        # 5 -> 1 EOS backpack + 2 normal backpacks + 2 items
        # 2 -> features
        self.assertEqual(actual_state.shape, (2,5,2))

        expected_state = [
            [
                [0, 0],  # EOS Backpack
                [10, 0], # Backpack 1
                [10, 0], # Backpack 2
                [5, 3],  # Item 0
                [5, 3]   # Item 1
            ],
            [
                [0, 0],  # EOS Backpack
                [10, 0], # Backpack 1
                [10, 0], # Backpack 2
                [5, 3],  # Item 0
                [5, 3]   # Item 1
            ]
        ]

        self.assertTrue(np.all(actual_state == expected_state))

    def test_compute_masks_fn(self):
        env = Knapsack('knapsack', {
            "batch_size": 2,
            "num_items": 2,
            "num_backpacks": 2,

            "min_item_value": 3,
            "max_item_value": 3,

            "min_item_weight": 5,
            "max_item_weight": 5,

            "min_backpack_capacity": 10,
            "max_backpack_capacity": 10
        })

        actual_backpacks_masks, actual_items_masks = env.compute_masks()
        
        # Test masks shape
        self.assertEqual(actual_backpacks_masks.shape, (2,5))
        self.assertEqual(actual_items_masks.shape, (2,5))

        # Test the actual values of backpack mask
        expected_backpacks_masks = [
            [0, 0, 0, 1, 1],
            [0, 0, 0, 1, 1]
        ]
        self.assertTrue(np.all(actual_backpacks_masks == expected_backpacks_masks))

        # Test the actual values of backpack mask
        expected_item_masks = [
            [1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0]
        ]
        self.assertTrue(np.all(actual_items_masks == expected_item_masks))
    
    def test_step_fn(self):
        env = Knapsack('knapsack', {
            "batch_size": 2,
            "num_items": 2,
            "num_backpacks": 2,

            "min_item_value": 3,
            "max_item_value": 3,

            "min_item_weight": 5,
            "max_item_weight": 5,

            "min_backpack_capacity": 10,
            "max_backpack_capacity": 10
        })

        action = [
            [1, 3], # Problem 0: Place item 0 (index 3) at backpack 1 (index 1)
            [2, 4]  # Problem 1: Place item 1 (index 4) at backpack 2 (index 2)
        ]

        actual_state, actual_rewards, actual_dones, info =  env.step(action)

        # Test data shapes
        self.assertEqual(actual_state.shape, (2,5,2))
        self.assertEqual(actual_rewards.shape, (2,1))
        self.assertEqual(actual_dones.shape, (2,1))
        self.assertEqual(info['backpack_net_mask'].shape, (2,5))
        self.assertEqual(info['item_net_mask'].shape, (2,5))
        self.assertEqual(info['num_items_to_place'], 2)

        expected_state = [
            [
                [0, 0],  # 
                [10, 3], # Backpack's value is equal to the selected item
                [10, 0], # 
                [5, 3],  # Item selected
                [5, 3]   # 
            ],
            [
                [0, 0],  # 
                [10, 0], # 
                [10, 3], # Backpack's value is equal to the selected item
                [5, 3],  # 
                [5, 3]   # Item selected
            ]
        ]
        self.assertTrue(np.all(actual_state == expected_state))

        expected_rewards = [
            [3],
            [3]
        ]
        self.assertTrue(np.all(actual_rewards == expected_rewards))

        expected_dones = [
            [False],
            [False]
        ]
        self.assertTrue(np.all(actual_dones == expected_dones))


        # Test the actual values of backpack mask
        expected_backpacks_masks = [
            [0, 0, 0, 1, 1],
            [0, 0, 0, 1, 1]
        ]
        self.assertTrue(np.all(info['backpack_net_mask'] == expected_backpacks_masks))

        # Test the actual values of backpack mask
        expected_item_masks = [
            [1, 1, 1, 1, 0],
            [1, 1, 1, 0, 1]
        ]
        self.assertTrue(np.all(info['item_net_mask'] == expected_item_masks))

    def test_multiple_steps_fn(self):
        env = Knapsack('knapsack', {
            "batch_size": 2,
            "num_items": 2,
            "num_backpacks": 2,

            "min_item_value": 3,
            "max_item_value": 3,

            "min_item_weight": 5,
            "max_item_weight": 5,

            "min_backpack_capacity": 10,
            "max_backpack_capacity": 10
        })

        action_list = np.array([
            # 1 Step
            [
                [1, 3], # Problem 0: Place item 0 (index 3) at backpack 1 (index 1)
                [2, 4]  # Problem 1: Place item 1 (index 4) at backpack 2 (index 2)
            ],
            # 2 Step
            [
                [2, 4], # Problem 0: Place item 0 (index 3) at EOS backpack (index 0)
                [0, 3]  # Problem 1: Place item 1 (index 4) at EOS backpack (index 0)
            ]
        ])

        actual_state, actual_rewards, actual_dones, info =  env.multiple_steps(action_list)

        # Test data shapes
        self.assertEqual(actual_state.shape, (2,5,2))
        self.assertEqual(actual_rewards.shape, (2,2))
        self.assertEqual(actual_dones.shape, (2,2))
        self.assertEqual(info['backpack_net_mask'].shape, (2,5))
        self.assertEqual(info['item_net_mask'].shape, (2,5))
        self.assertEqual(info['num_items_to_place'], 2)

        expected_state = [
            [
                [0, 0],  # 
                [10, 3], # Step 1: Backpack's value is equal to the selected item
                [10, 3], # Step 2: Backpack's value is equal to the selected item
                [5, 3],  # Step 1: Item selected
                [5, 3]   # Step 2: Item selected
            ],
            [
                [0, 0],  # Step 2: Backpack's value is equal to the selected item
                [10, 0], # 
                [10, 3], # Step 1: Backpack's value is equal to the selected item
                [5, 3],  # Step 2: Item selected
                [5, 3]   # Step 1: Item selected
            ]
        ]

        self.assertTrue(np.all(actual_state == expected_state))

        expected_rewards = [
            [
                3.0, # Prob 1: Step 1
                3.0  # Prob 1: Step 2
            ],
            [
                3.0, # Prob 2: Step 1
                0    # Prob 2: Step 1
            ]
        ]
        self.assertTrue(np.all(actual_rewards == expected_rewards))

        expected_dones = [
            [
                0, # Prob 1: Step 1: Not Done -> 1 is still pending
                1  # Prob 1: Step 2: Done -> All items are placed
            ],
            [
                0, # Prob 1: Step 1: Not Done -> 1 is still pending
                1  # Prob 1: Step 2: Done -> All items are placed
            ]
        ]
        self.assertTrue(np.all(actual_dones == expected_dones))


        # Test the actual values of backpack mask
        expected_backpacks_masks = [
            [0, 0, 0, 1, 1],
            [0, 0, 0, 1, 1]
        ]
        self.assertTrue(np.all(info['backpack_net_mask'] == expected_backpacks_masks))

        # Test the actual values of backpack mask
        expected_item_masks = [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1]
        ]
        self.assertTrue(np.all(info['item_net_mask'] == expected_item_masks))
