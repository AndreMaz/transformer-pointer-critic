import sys
sys.path.append('.')

import unittest
import numpy as np

### Custom Imports
from environment.custom.knapsack.item import Item

class TestItem(unittest.TestCase):

    # def test_upper(self):
    #     self.assertEqual('foo'.upper(), 'FOO')

    def test_constructor(self):
        item = Item(0, 1, 1, 2, 2)  # Light item

        self.assertEqual(item.id, 0)
        self.assertEqual(item.value, 1)
        self.assertEqual(item.weight, 2)
        self.assertEqual(item.taken, False)

    def test_methods(self):
        item = Item(0, 1, 1, 2, 2)  # Light item

        self.assertEqual(item.is_taken(), False)
        item.take()
        self.assertEqual(item.is_taken(), True)

        # Can't take the item that was already taken
        with self.assertRaises(ValueError):
            item.take()

        item.place_back()

        self.assertEqual(item.is_taken(), False)
