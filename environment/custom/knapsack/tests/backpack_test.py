import sys
sys.path.append('.')

import unittest
import numpy as np

### Custom Imports
from environment.custom.knapsack.backpack import Backpack, EOS_BACKPACK, NORMAL_BACKPACK
from environment.custom.knapsack.item import Item

class TestBackpack(unittest.TestCase):

    # def test_upper(self):
    #     self.assertEqual('foo'.upper(), 'FOO')

    def test_constructor(self):
        backpack = Backpack(0, EOS_BACKPACK, 10, 10)

        self.assertEqual(backpack.id, 0)
        self.assertEqual(backpack.type, EOS_BACKPACK)
        self.assertEqual(backpack.capacity, 10)

        self.assertEqual(backpack.current_capacity, 0)
        self.assertEqual(backpack.current_value, 0)
        self.assertEqual(len(backpack.items), 0)

    def test_add_item_backpack(self):
        backpack = Backpack(0, NORMAL_BACKPACK, 10, 10)

        item1 = Item(0, 1, 1, 2, 2)  # Light item
        item2 = Item(1, 100, 100, 1, 1)  # Light but valuable item
        item3 = Item(1, 1, 1, 100, 100)  # Heavy item

        is_valid, capacity_diff, current_value = backpack.add_item(item1)
        self.assertEqual(is_valid, True)
        self.assertEqual(capacity_diff, 8)  # 10 (Backpack) - 2 (item1)
        self.assertEqual(current_value, 1)  # 1 (item1)

        is_valid, capacity_diff, current_value = backpack.add_item(item2)
        self.assertEqual(is_valid, True)
        # 10 (Backpack) - ( 2 (item1) + 1 (item2))
        self.assertEqual(capacity_diff, 7)
        self.assertEqual(current_value, 101)  # 1 (item1) + 100 (item2)

        is_valid, capacity_diff, current_value = backpack.add_item(item3)
        self.assertEqual(is_valid, False)
        # 10 (Backpack) - ( 2 (item1) + 1 (item2) + 100 (item3) )
        self.assertEqual(capacity_diff, 93)
        # 1 (item1) + 100 (item2) + 1 (item3)
        self.assertEqual(current_value, 102)

        # Item is taken even if it's placed at EOS backpack
        self.assertEqual(item1.is_taken(), True)
        self.assertEqual(item2.is_taken(), True)
        self.assertEqual(item3.is_taken(), True)

        # Must have 3 items stored
        self.assertEqual(len(backpack.items), 3)

        self.assertEqual(backpack.is_full(), True)

        # Clear the backpack
        backpack.clear()

        self.assertEqual(backpack.current_capacity, 0)
        self.assertEqual(backpack.current_value, 0)
        self.assertEqual(len(backpack.items), 0)

        # Item's returned back
        self.assertEqual(item1.is_taken(), False)
        self.assertEqual(item2.is_taken(), False)
        self.assertEqual(item3.is_taken(), False)
