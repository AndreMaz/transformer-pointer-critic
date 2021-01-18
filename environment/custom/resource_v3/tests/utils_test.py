import sys
sys.path.append('.')
import unittest
import numpy as np


from environment.custom.resource_v3.utils import round_half_up

class TestResource(unittest.TestCase):
    def test_round_ops(self):
        expected_total = 0.41
        total = 0.4099999964237213

        expected_current = 0.35
        current = 0.34999999962747097

        expected_req_load = 0.06
        req_load = 0.05999999865889549

    
        result = total >= current + req_load

        self.assertFalse(result)

        rounded_result = round_half_up(total, 2) >= round_half_up(current, 2) + round_half_up(req_load, 2)
        self.assertEqual(round_half_up(total, 2), expected_total)
        self.assertEqual(round_half_up(current, 2), expected_current)
        self.assertEqual(round_half_up(req_load, 2), expected_req_load)