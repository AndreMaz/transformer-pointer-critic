import sys
sys.path.append('.')
import unittest
import numpy as np

# Custom Imports
from environment.custom.resource_v3.heuristic.factory import DominantResourceHeuristic, RandomHeuristic, CPLEXSolver
from environment.custom.resource_v3.heuristic.dominant_heuristic import compute_potential_placement_diffs
from environment.custom.resource_v3.misc.utils import compute_stats

class TestDominantHeuristic(unittest.TestCase):
    def setUp(self) -> None:
    
        heuristic_opts = {
                "resource_sort_descending": True,
                "node_sort_descending": True
            }

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
        normalization_factor = 100
        
        self.solver = DominantResourceHeuristic(
            node_sample_size,
            normalization_factor,
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

        # First node that's not EOS ||   [ 1,  2,  3]
        dominant_resource = node_list[1].compute_dominant_resource(
            resource_list[0] # [ 2,  1,  4]
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

    def test_compute_potential_placement_diffs(self):
        node_list = self.solver.parse_nodes(self.dummy_state)
        EOS_NODE = node_list.pop(0)
        resource_list = self.solver.parse_resources(self.dummy_state)

        actual = compute_potential_placement_diffs(resource_list[0], node_list)

        self.assertEqual(actual[0][0], -1)
        self.assertEqual(actual[1][0], 1)

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

class TestRandomHeuristic(unittest.TestCase):
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
        normalization_factor = 100
        
        self.solver = RandomHeuristic(
            node_sample_size,
            normalization_factor,
            heuristic_opts
        )
    
    def test_solver(self):
        self.solver.solve(self.dummy_state)

        delta,\
        num_rejected,\
        empty_nodes = compute_stats(self.solver.solution)

        self.assertEqual(
            delta, 1
        )

        self.assertEqual(
            num_rejected, 1
        )

        self.assertEqual(
            empty_nodes, 1
        )

class TestCPLEXSolver(unittest.TestCase):
    def setUp(self) -> None:
    
        heuristic_opts = {
            "use": False,
            "time_limit_ms": 60000,
            "greedy_with_critical_resource": True,
            "num_threads": 4
        }

        normalization_factor = 100
        node_sample_size = 3
        
        self.solver = CPLEXSolver(
            node_sample_size,
            normalization_factor,
            heuristic_opts
        )

    def test_solver_SHOULD_reject_2(self):
        dummy_state = np.array([
            [
                # Nodes
                # CPU   RAM  MEM
                [-2.0, -2.0, -2.0],
                [0.1,  0.2,  0.3],
                [0.5,  0.2,  0.6],

                # Resources
                # CPU  RAM   MEM
                [0.2,  0.1,  0.9], # Should Reject. Too Big
                [0.3,  0.5,  0.8], # Should Reject. Too Big
            ]
        ], dtype='float32')
        self.solver.solve(dummy_state)

        delta,\
        num_rejected,\
        empty_nodes = compute_stats(self.solver.solution)

        self.assertEqual(
            delta, 0.1
        )

        self.assertEqual(
            num_rejected, 2
        )

        self.assertEqual(
            empty_nodes, 2
        )


    def test_solver_SHOULD_reject_1(self):
        dummy_state = np.array([
            [
                # Nodes
                # CPU   RAM  MEM
                [-2.0, -2.0, -2.0],
                [0.1,  0.2,  0.3],
                [0.5,  0.3,  0.6],

                # Resources
                # CPU  RAM   MEM
                [0.2,  0.1,  0.4],
                [0.3,  0.5,  0.8], # Should Reject. Too Big
            ]
        ], dtype='float32')
        self.solver.solve(dummy_state)

        delta,\
        num_rejected,\
        empty_nodes = compute_stats(self.solver.solution)

        self.assertEqual(
            delta, 0.1
        )

        self.assertEqual(
            num_rejected, 1
        )

        self.assertEqual(
            empty_nodes, 1
        )
    
    def test_solver_SHOULD_place_2_at_2_nodes(self):
        dummy_state = np.array([
            [
                # Nodes
                # CPU   RAM  MEM
                [-2.0, -2.0, -2.0],
                [0.3,  0.6,  0.6],
                [0.5,  0.8,  0.7],

                # Resources
                # CPU  RAM   MEM
                [0.2,  0.5,  0.4], 
                [0.3,  0.5,  0.4], 
            ]
        ], dtype='float32')
        self.solver.solve(dummy_state)

        delta,\
        num_rejected,\
        empty_nodes = compute_stats(self.solver.solution)

        self.assertEqual(
            delta, 0.1
        )

        self.assertEqual(
            num_rejected, 0
        )

        self.assertEqual(
            empty_nodes, 0
        )

    def test_solver_SHOULD_place_2_at_1_node(self):
        dummy_state = np.array([
            [
                # Nodes
                # CPU   RAM  MEM
                [-2.0, -2.0, -2.0],
                [0.5,  1.0,  0.9], # Should Place both reqs here
                [0.1,  0.1,  0.1],

                # Resources
                # CPU  RAM   MEM
                [0.2,  0.5,  0.4], 
                [0.3,  0.5,  0.4], 
            ]
        ], dtype='float32')
        self.solver.solve(dummy_state)

        delta,\
        num_rejected,\
        empty_nodes = compute_stats(self.solver.solution)

        self.assertEqual(
            delta, 0.0
        )

        self.assertEqual(
            num_rejected, 0
        )

        self.assertEqual(
            empty_nodes, 1
        )
