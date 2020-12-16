import sys
sys.path.append('.')

import unittest
import numpy as np

### Custom Imports
from environment.custom.resource.penalty import GreedyPenalty


class TestItem(unittest.TestCase):

    def setUp(self) -> None:
        opts = {
                "CPU_misplace_penalty": 5,
                "RAM_misplace_penalty": 10,
                "MEM_misplace_penalty": 15
        }

        EOS_CODE = -1
        resource_normalization_factor = 1

        self.penalizer = GreedyPenalty(
            opts, EOS_CODE, resource_normalization_factor    
        )
    
    def test_constructor(self):
        self.assertEqual(self.penalizer.CPU_penalty, 5)
        self.assertEqual(self.penalizer.RAM_penalty, 10)
        self.assertEqual(self.penalizer.MEM_penalty, 15)
    
    def test_compute_CPU_penalty(self):
        resource_CPU = 3
        expected_CPU = 8
        self.assertEqual(self.penalizer.compute_CPU_penalty(resource_CPU), expected_CPU)
    
    def test_compute_RAM_penalty(self):
        resource_RAM = 7
        expected_RAM = 17
        self.assertEqual(self.penalizer.compute_RAM_penalty(resource_RAM), expected_RAM)

    def test_compute_MEM_penalty(self):
        resource_MEM = 1
        expected_MEM = 16
        self.assertEqual(self.penalizer.compute_MEM_penalty(resource_MEM), expected_MEM)

    def test_compute_penalty(self):
        resource_CPU = 5
        resource_RAM = 10
        resource_MEM = 10

        expected_CPU = 10
        expected_RAM = 20
        expected_MEM = 25

        actual_CPU, actual_RAM, actual_MEM = self.penalizer.compute_penalty(
            resource_CPU, resource_RAM, resource_MEM
        )

        self.assertEqual(expected_CPU, actual_CPU)
        self.assertEqual(expected_RAM, actual_RAM)
        self.assertEqual(expected_MEM, actual_MEM)

    def test_should_NOT_to_penalize(self):
        bin_lower_type = 2
        bin_upper_type = 5
        
        resource_type = 3

        self.assertEqual(
            self.penalizer.to_penalize(bin_lower_type, bin_upper_type, resource_type),
            False
        )
    
    def test_should_to_penalize_lower_bound(self):
        bin_lower_type = 2
        bin_upper_type = 5
        
        resource_type = 1

        self.assertEqual(
            self.penalizer.to_penalize(bin_lower_type, bin_upper_type, resource_type),
            True
        )
    
    def test_should_to_penalize_upper_bound(self):
        bin_lower_type = 2
        bin_upper_type = 5
        
        resource_type = 6

        self.assertEqual(
            self.penalizer.to_penalize(bin_lower_type, bin_upper_type, resource_type),
            True
        )

    def test_batch_should_NOT_penalize(self):
        bin_lower_type = np.array([
            0, # node 0 lower bound
            4, # node 1 lower bound
        ], dtype="float32")

        bin_upper_type = np.array([
            7, # node 0 lower bound
            6, # node 1 lower bound
        ], dtype="float32")

        resource_type = np.array([
            5, # resource 0 type
            6, # resource 1 type
        ], dtype="float32")


        expected_result = [0, 0]

        actual_result = self.penalizer.to_penalize_batch(
            bin_lower_type,
            bin_upper_type,
            resource_type
        )

        self.assertEqual(actual_result.numpy().tolist(), expected_result)

    def test_batch_should_penalize_upper_bound(self):
        bin_lower_type = np.array([
            0, # node 0 lower bound
            4, # node 1 lower bound
        ], dtype="float32")

        bin_upper_type = np.array([
            7, # node 0 lower bound
            6, # node 1 lower bound
        ], dtype="float32")

        resource_type = np.array([
            55, # resource 0 type
            65, # resource 1 type
        ], dtype="float32")


        expected_result = [1, 1]

        actual_result = self.penalizer.to_penalize_batch(
            bin_lower_type,
            bin_upper_type,
            resource_type
        )

        self.assertEqual(actual_result.numpy().tolist(), expected_result)

    def test_batch_should_penalize_lower_bound(self):
        bin_lower_type = np.array([
            3, # node 0 lower bound
            4, # node 1 lower bound
        ], dtype="float32")

        bin_upper_type = np.array([
            7, # node 0 lower bound
            6, # node 1 lower bound
        ], dtype="float32")

        resource_type = np.array([
            1, # resource 0 type
            1, # resource 1 type
        ], dtype="float32")


        expected_result = [1, 1]

        actual_result = self.penalizer.to_penalize_batch(
            bin_lower_type,
            bin_upper_type,
            resource_type
        )

        self.assertEqual(actual_result.numpy().tolist(), expected_result)

    def test_batch_should_penalize_first_element(self):
        bin_lower_type = np.array([
            3, # node 0 lower bound
            4, # node 1 lower bound
        ], dtype="float32")

        bin_upper_type = np.array([
            7, # node 0 lower bound
            6, # node 1 lower bound
        ], dtype="float32")

        resource_type = np.array([
            199, # resource 0 type
            5, # resource 1 type
        ], dtype="float32")


        expected_result = [1, 0]

        actual_result = self.penalizer.to_penalize_batch(
            bin_lower_type,
            bin_upper_type,
            resource_type
        )

        self.assertEqual(actual_result.numpy().tolist(), expected_result)

    def test_batch_should_penalize_second_element(self):
        bin_lower_type = np.array([
            3, # node 0 lower bound
            4, # node 1 lower bound
        ], dtype="float32")

        bin_upper_type = np.array([
            7, # node 0 lower bound
            6, # node 1 lower bound
        ], dtype="float32")

        resource_type = np.array([
            4, # resource 0 type
            588, # resource 1 type
        ], dtype="float32")


        expected_result = [0, 1]

        actual_result = self.penalizer.to_penalize_batch(
            bin_lower_type,
            bin_upper_type,
            resource_type
        )

        self.assertEqual(actual_result.numpy().tolist(), expected_result)

