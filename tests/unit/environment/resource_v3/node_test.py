import sys
sys.path.append('.')
import unittest
import numpy as np

from environment.custom.resource_v3.node import Node
from environment.custom.resource_v3.resource import Resource
from environment.custom.resource_v3.utils import round_half_up

class TestNode(unittest.TestCase):
    def setUp(self) -> None:
        batch_id = 0
        id = 0 # EOS NODE
        self.node_representation = np.array([0.9, 0.8, 0.7], dtype="float32")
        
        self.node = Node(batch_id, id, self.node_representation)

    def test_constructor(self):
        self.assertEqual(self.node.batch_id, 0)
        self.assertEqual(self.node.id, 0)
        
        self.assertEqual(
            self.node.CPU.tolist(),
            [self.node_representation[0].tolist()]
        )

        self.assertEqual(
            self.node.RAM.tolist(),
            [self.node_representation[1].tolist()]
        )

        self.assertEqual(
            self.node.MEM.tolist(),
            [self.node_representation[2].tolist()]
        )

        self.assertEqual(len(self.node.req_list), 0)

        self.assertEqual(len(self.node.CPU_history), 1)
        self.assertEqual(len(self.node.RAM_history), 1)
        self.assertEqual(len(self.node.MEM_history), 1)

    def test_insert_req_EOS_node(self):
        batch_id = 0
        id = 1
        req_representation = np.array([0.3, 0.2, 0.1], dtype="float32")
        
        req = Resource(
            batch_id, id, req_representation
        )

        self.node.insert_req(req)

        self.assertEqual(len(self.node.req_list),1)

        # EOS Node is not updated
        self.assertEqual(
            self.node.remaining_CPU.tolist(), self.node_representation[0]
        )
        self.assertEqual(
            self.node.remaining_RAM.tolist(), self.node_representation[1]
        )
        self.assertEqual(
            self.node.remaining_MEM.tolist(), self.node_representation[2]
        )

    def test_insert_req_regular_node(self):
        batch_id = 0
        id = 1
        req_representation = np.array([0.3, 0.2, 0.1], dtype="float32")
        
        req = Resource(
            batch_id, id, req_representation
        )
        # Change the node ID to a non EOS
        self.node.id = 1

        self.node.insert_req(req)

        self.assertEqual(len(self.node.req_list),1)

        # EOS Node is not updated
        precision = 2
        self.assertAlmostEqual(
            self.node.remaining_CPU.tolist()[0],
            self.node_representation[0] - req_representation[0],
            precision
        )
        self.assertAlmostEqual(
            self.node.remaining_RAM.tolist()[0],
            self.node_representation[1] - req_representation[1],
            precision
        )
        self.assertAlmostEqual(
            self.node.remaining_MEM.tolist()[0],
            self.node_representation[2] - req_representation[2],
            precision
        )
    
    def test_reset(self):
        batch_id = 0
        id = 1
        req_representation = np.array([0.3, 0.2, 0.1], dtype="float32")
        req = Resource(
            batch_id, id, req_representation
        )

        # Change the node ID to a non EOS
        self.node.id = 1

        # Initially there's no reqs
        self.assertEqual(len(self.node.req_list),0)

        self.node.insert_req(req)

        # Should be one
        self.assertEqual(len(self.node.req_list),1)
        
        self.node.reset()

        # Reset. Back to zero
        self.assertEqual(len(self.node.req_list),0)

        # Back to original values
        self.assertEqual(
            self.node.remaining_CPU.tolist(), self.node_representation[0]
        )
        self.assertEqual(
            self.node.remaining_RAM.tolist(), self.node_representation[1]
        )
        self.assertEqual(
            self.node.remaining_MEM.tolist(), self.node_representation[2]
        )