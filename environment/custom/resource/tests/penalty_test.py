import sys
sys.path.append('.')

import unittest
import numpy as np

### Custom Imports
from environment.custom.resource.penalty import Penalty


class TestItem(unittest.TestCase):

    def setUp(self) -> None:
        CPU_misplace_penalty = np.array([5], dtype='float32')
        RAM_misplace_penalty = np.array([10], dtype='float32')
        MEM_misplace_penalty = np.array([15], dtype='float32')

        self.penalizer = Penalty(
            CPU_misplace_penalty,
            RAM_misplace_penalty,
            MEM_misplace_penalty
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