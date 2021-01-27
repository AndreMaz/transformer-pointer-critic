import sys
sys.path.append('.')
import unittest
import numpy as np

# Custom Imports
from environment.custom.resource_v3.env import ResourceEnvironmentV3
from environment.custom.resource_v3.heuristic import GreedyHeuristic, compute_dominant_resource, node_sorting_fn, resource_sorting_fn

class TestHeuristic(unittest.TestCase):
    def setUp(self) -> None:
    
        heuristic_opts = {}

        self.dummy_state = np.array([
            [
                [-2, -2, -2],
                [ 1,  2,  3],
                [ 5,  2,  6],
                [ 2,  1,  4],
                [ 3,  5,  8],
            ]
        ], dtype='float32')
        
        node_sample_size = 3
        
        self.solver = GreedyHeuristic(
            node_sample_size,
            heuristic_opts
        )

    def test_parse_nodes(self):
        expected_num_nodes = 3

        node_list = self.solver.parse_nodes(self.dummy_state)

        self.assertEqual(
            len(node_list),
            expected_num_nodes
        )

        expected_first_node = np.array(
            [-2, -2, -2,], dtype="float32"
        )
        self.assertEqual(
            node_list[0].get_tensor_rep().tolist(),
            expected_first_node.tolist()
        )

        expected_first_node = np.array(
            [1, 2, 3,], dtype="float32"
        )
        self.assertEqual(
            node_list[1].get_tensor_rep().tolist(),
            expected_first_node.tolist()
        )

        expected_first_node = np.array(
            [5, 2, 6,], dtype="float32"
        )
        self.assertEqual(
            node_list[2].get_tensor_rep().tolist(),
            expected_first_node.tolist()
        )
    
    def test_parse_resources(self):
        expected_num_resources = 2

        resource_list = self.solver.parse_resources(self.dummy_state)

        self.assertEqual(
            len(resource_list),
            expected_num_resources
        )

        expected_first_node = np.array(
            [2, 1, 4,], dtype="float32"
        )
        self.assertEqual(
            resource_list[0].get_tensor_rep().tolist(),
            expected_first_node.tolist()
        )

        expected_first_node = np.array(
            [3, 5, 8,], dtype="float32"
        )
        self.assertEqual(
            resource_list[1].get_tensor_rep().tolist(),
            expected_first_node.tolist()
        )

    def test_compute_dominant_resource(self):
        node_list = self.solver.parse_nodes(self.dummy_state)
        resource_list = self.solver.parse_resources(self.dummy_state)

        dominant_resource = compute_dominant_resource(
            node_list[1], # First node that's node EOS ||   [ 1,  2,  3]
            resource_list[0] #                              [ 2,  1,  4]
        )

        expected_dominant = np.array([-1], dtype='float32')

        self.assertEqual(
            dominant_resource.tolist(),
            expected_dominant.tolist()
        )

    def test_place_single_resource(self):
        node_list = self.solver.parse_nodes(self.dummy_state)
        EOS_NODE = node_list.pop(0)
        resource_list = self.solver.parse_resources(self.dummy_state)

        self.solver.place_single_resource(
            resource_list[0],
            node_list,
            EOS_NODE
        )

        node_list = [EOS_NODE] + node_list

        expected_num_reqs_at_node0 = 0
        self.assertEqual(
            len(node_list[0].req_list),
            expected_num_reqs_at_node0
        )

        expected_num_reqs_at_node1 = 0
        self.assertEqual(
            len(node_list[1].req_list),
            expected_num_reqs_at_node1
        )

        expected_num_reqs_at_node2 = 1
        self.assertEqual(
            len(node_list[2].req_list),
            expected_num_reqs_at_node2
        )
    
    def test_solver(self):
        
        self.solver.solve(self.dummy_state)

        expected_num_reqs_at_node0 = 1
        self.assertEqual(
            len(self.solver.solution[0].req_list),
            expected_num_reqs_at_node0
        )

        actual_resource_at_node0 = self.solver.solution[0].req_list[0]
        expected_resource_at_node0 = np.array([3, 5, 8], dtype="float32")

        self.assertEqual(
            actual_resource_at_node0.get_tensor_rep().tolist(),
            expected_resource_at_node0.tolist()
        )

        expected_num_reqs_at_node1 = 0
        self.assertEqual(
            len(self.solver.solution[1].req_list),
            expected_num_reqs_at_node1
        )

        expected_num_reqs_at_node2 = 1
        self.assertEqual(
            len(self.solver.solution[2].req_list),
            expected_num_reqs_at_node2
        )
        actual_resource_at_node2 = self.solver.solution[2].req_list[0]
        expected_resource_at_node2 = np.array([2, 1, 4], dtype="float32")
        
        self.assertEqual(
            actual_resource_at_node2.get_tensor_rep().tolist(),
            expected_resource_at_node2.tolist()
        )