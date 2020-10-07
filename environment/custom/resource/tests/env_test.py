import numpy as np
import unittest
import sys
sys.path.append('.')

# Custom Imports
from environment.custom.resource.env import ResourceEnvironment
from environment.custom.resource.reward import Reward
from environment.custom.resource.penalty import Penalty

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

        "normalization_factor": 1,

        "num_user_levels": 1, 
        "reward_per_level": [ 10, 20 ],
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

class TestItem(unittest.TestCase):

    def setUp(self) -> None:
        self.env = ResourceEnvironment('Resource', ENV_CONFIG)

    def test_constructor(self):
        self.env.name = 'Resource'