import sys
sys.path.append('.')
import unittest
import numpy as np

from environment.custom.knapsack_v2.bin import Bin
from environment.custom.knapsack_v2.item import Item


class TestBin(unittest.TestCase):
    def setUp(self) -> None:
        batch_id = 0
        id = 0 # EOS NODE
        self.node_representation = np.array([0.9, 0.0], dtype="float32")
        
        self.node = Bin(batch_id, id, self.node_representation)

    def test_constructor(self):
        self.assertEqual(self.node.batch_id, 0)
        self.assertEqual(self.node.id, 0)
        
        self.assertEqual(
            self.node.capacity.tolist(),
            [self.node_representation[0].tolist()]
        )

        self.assertEqual(
            self.node.current_load.tolist(),
            [self.node_representation[1].tolist()]
        )

        self.assertEqual(len(self.node.item_list), 0)

        self.assertEqual(len(self.node.load_history), 1)
    
        self.assertEqual(
            self.node.load_history[0].tolist(),
            [self.node_representation[1].tolist()]
        )
    
    def test_insert_req_EOS_node(self):
        batch_id = 0
        id = 1
        req_representation = np.array([0.3, 0.2], dtype="float32")
        
        req = Item(
            batch_id, id, req_representation
        )

        self.node.insert_item(req)

        self.assertEqual(len(self.node.item_list),1)

        # EOS Bin is not updated
        self.assertEqual(
            self.node.capacity.tolist(), self.node_representation[0]
        )
        self.assertEqual(
            self.node.current_load.tolist(), self.node_representation[1]
        )
    
    def test_insert_req_regular_node(self):
        batch_id = 0
        id = 1
        req_representation = np.array([0.3, 0.2], dtype="float32")
        
        req = Item(
            batch_id, id, req_representation
        )
        # Change the node ID to a non EOS
        self.node.id = 1

        self.node.insert_item(req)

        self.assertEqual(len(self.node.item_list),1)

        # EOS Node is not updated
        precision = 2
        self.assertAlmostEqual(
            self.node.capacity.tolist()[0],
            self.node_representation[0],
            precision
        )
        self.assertAlmostEqual(
            self.node.current_load.tolist()[0],
            self.node_representation[1] + req_representation[0],
            precision
        )
    
    def test_reset(self):
        batch_id = 0
        id = 1
        req_representation = np.array([0.3, 0.2], dtype="float32")
        req = Item(
            batch_id, id, req_representation
        )

        # Change the node ID to a non EOS
        self.node.id = 1

        # Initially there's no reqs
        self.assertEqual(len(self.node.item_list),0)

        self.node.insert_item(req)

        # Should be one
        self.assertEqual(len(self.node.item_list),1)
        
        self.node.reset()

        # Reset. Back to zero
        self.assertEqual(len(self.node.item_list),0)

        # Back to original values
        self.assertEqual(
            self.node.capacity.tolist(), self.node_representation[0]
        )
        self.assertEqual(
            self.node.current_load.tolist(), self.node_representation[1]
        )