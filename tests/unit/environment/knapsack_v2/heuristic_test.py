import sys
sys.path.append('.')
import unittest
import numpy as np

from environment.custom.knapsack_v2.heuristic.factory import WasteReductionHeuristic, ORTools, RandomHeuristic
from environment.custom.knapsack_v2.misc.utils import compute_stats

class TestORToolsHeuristics(unittest.TestCase):
    def setUp(self) -> None:
    
        heuristic_opts = {
                "time_limit_ms": 1000,
                "num_threads": 1
            }

        self.dummy_state = np.array([
            [
                [-2.0, -2.0],
                [0.1,  0.0],
                [0.5,  0.0],
                [0.2,  0.1],
                [0.3,  0.5],
                [0.1,  0.4],
                [0.9,  0.4],
            ]
        ], dtype='float32')
        
        node_sample_size = 3
        
        self.solver = ORTools(
            node_sample_size,
            heuristic_opts
        )
    
    def test_parse_bins(self):
        expected_num_bin = 3

        bin_list = self.solver.parse_bins(self.dummy_state)

        self.assertEqual(
            len(bin_list),
            expected_num_bin
        )

        expected_first_node = np.array(
            [-2, -2], dtype="float32"
        )
        self.assertEqual(
            bin_list[0].get_tensor_rep().tolist(),
            expected_first_node.tolist()
        )

        expected_first_node = np.array(
            [0.1, 0], dtype="float32"
        )
        self.assertEqual(
            bin_list[1].get_tensor_rep().tolist(),
            expected_first_node.tolist()
        )

        expected_first_node = np.array(
            [0.5, 0], dtype="float32"
        )
        self.assertEqual(
            bin_list[2].get_tensor_rep().tolist(),
            expected_first_node.tolist()
        )

    def test_parse_resources(self):
        expected_num_resources = 4

        resource_list = self.solver.parse_items(self.dummy_state)

        self.assertEqual(
            len(resource_list),
            expected_num_resources
        )

        expected_first_item = np.array(
            [0.2, 0.1], dtype="float32"
        )
        self.assertEqual(
            resource_list[0].get_tensor_rep().tolist(),
            expected_first_item.tolist()
        )

        expected_first_item = np.array(
            [0.3, 0.5,], dtype="float32"
        )
        self.assertEqual(
            resource_list[1].get_tensor_rep().tolist(),
            expected_first_item.tolist()
        )

        expected_third_item = np.array(
            [0.1, 0.4,], dtype="float32"
        )
        self.assertEqual(
            resource_list[2].get_tensor_rep().tolist(),
            expected_third_item.tolist()
        )

        expected_fourth_item = np.array(
            [0.9, 0.4,], dtype="float32"
        )
        self.assertEqual(
            resource_list[3].get_tensor_rep().tolist(),
            expected_fourth_item.tolist()
        )
    
    def test_solver(self):
        self.solver.solve(self.dummy_state)


        actual_reward,\
        actual_empty_nodes,\
        actual_num_rejected_items,\
        actual_rejected_value = compute_stats(self.solver.solution)
        
        precision = 2
        self.assertAlmostEqual(
            actual_reward, 1.0, precision
        )

        self.assertEqual(
            actual_empty_nodes, 0
        )

        self.assertEqual(
            actual_num_rejected_items, 1
        )

        self.assertEqual(
            actual_rejected_value, 0.4
        )

        self.assertEqual(
            len(self.solver.solution[0].item_list),
            1
        )

        self.assertEqual(
            len(self.solver.solution[1].item_list),
            1
        )

        self.assertEqual(
            len(self.solver.solution[2].item_list),
            2
        )

class TestWasteReductionHeuristic(unittest.TestCase):
    def setUp(self) -> None:
    
        heuristic_opts = {
                "item_sort_descending": True,
                "bin_sort_descending": False
            }

        self.dummy_state = np.array([
            [
                [-2.0, -2.0],
                [0.1,  0.0],
                [0.5,  0.0],
                [0.2,  0.1],
                [0.3,  0.5],
                [0.1,  0.4],
                [0.9,  0.4],
            ]
        ], dtype='float32')
        
        node_sample_size = 3
        
        self.solver = WasteReductionHeuristic(
            node_sample_size,
            heuristic_opts
        )
    
    def test_parse_bins(self):
        expected_num_bin = 3

        bin_list = self.solver.parse_bins(self.dummy_state)

        self.assertEqual(
            len(bin_list),
            expected_num_bin
        )

        expected_first_node = np.array(
            [-2, -2], dtype="float32"
        )
        self.assertEqual(
            bin_list[0].get_tensor_rep().tolist(),
            expected_first_node.tolist()
        )

        expected_first_node = np.array(
            [0.1, 0], dtype="float32"
        )
        self.assertEqual(
            bin_list[1].get_tensor_rep().tolist(),
            expected_first_node.tolist()
        )

        expected_first_node = np.array(
            [0.5, 0], dtype="float32"
        )
        self.assertEqual(
            bin_list[2].get_tensor_rep().tolist(),
            expected_first_node.tolist()
        )

    def test_parse_resources(self):
        expected_num_resources = 4

        resource_list = self.solver.parse_items(self.dummy_state)

        self.assertEqual(
            len(resource_list),
            expected_num_resources
        )

        expected_first_item = np.array(
            [0.2, 0.1], dtype="float32"
        )
        self.assertEqual(
            resource_list[0].get_tensor_rep().tolist(),
            expected_first_item.tolist()
        )

        expected_first_item = np.array(
            [0.3, 0.5,], dtype="float32"
        )
        self.assertEqual(
            resource_list[1].get_tensor_rep().tolist(),
            expected_first_item.tolist()
        )

        expected_third_item = np.array(
            [0.1, 0.4,], dtype="float32"
        )
        self.assertEqual(
            resource_list[2].get_tensor_rep().tolist(),
            expected_third_item.tolist()
        )

        expected_fourth_item = np.array(
            [0.9, 0.4,], dtype="float32"
        )
        self.assertEqual(
            resource_list[3].get_tensor_rep().tolist(),
            expected_fourth_item.tolist()
        )
    
    def test_solver(self):
        self.solver.solve(self.dummy_state)


        actual_reward,\
        actual_empty_nodes,\
        actual_num_rejected_items,\
        actual_rejected_value = compute_stats(self.solver.solution)
        
        precision = 2
        self.assertAlmostEqual(
            actual_reward[0], 1.0, precision
        )

        self.assertEqual(
            actual_empty_nodes, 0
        )

        self.assertEqual(
            actual_num_rejected_items, 1
        )

        self.assertEqual(
            actual_rejected_value, 0.4
        )

        self.assertEqual(
            len(self.solver.solution[0].item_list),
            1
        )

        self.assertEqual(
            len(self.solver.solution[1].item_list),
            1
        )

        self.assertEqual(
            len(self.solver.solution[2].item_list),
            2
        )

class TestRandomHeuristic(unittest.TestCase):
    def setUp(self) -> None:
    
        heuristic_opts = {}

        self.dummy_state = np.array([
            [
                [-2.0, -2.0],
                [0.1,  0.0],
                [0.5,  0.0],
                [0.2,  0.1],
                [0.3,  0.5],
                [0.1,  0.4],
                [0.9,  0.4],
            ]
        ], dtype='float32')
        
        node_sample_size = 3
        
        self.solver = RandomHeuristic(
            node_sample_size,
            heuristic_opts
        )
    
    def test_parse_bins(self):
        expected_num_bin = 3

        bin_list = self.solver.parse_bins(self.dummy_state)

        self.assertEqual(
            len(bin_list),
            expected_num_bin
        )

        expected_first_node = np.array(
            [-2, -2], dtype="float32"
        )
        self.assertEqual(
            bin_list[0].get_tensor_rep().tolist(),
            expected_first_node.tolist()
        )

        expected_first_node = np.array(
            [0.1, 0], dtype="float32"
        )
        self.assertEqual(
            bin_list[1].get_tensor_rep().tolist(),
            expected_first_node.tolist()
        )

        expected_first_node = np.array(
            [0.5, 0], dtype="float32"
        )
        self.assertEqual(
            bin_list[2].get_tensor_rep().tolist(),
            expected_first_node.tolist()
        )

    def test_parse_resources(self):
        expected_num_resources = 4

        resource_list = self.solver.parse_items(self.dummy_state)

        self.assertEqual(
            len(resource_list),
            expected_num_resources
        )

        expected_first_item = np.array(
            [0.2, 0.1], dtype="float32"
        )
        self.assertEqual(
            resource_list[0].get_tensor_rep().tolist(),
            expected_first_item.tolist()
        )

        expected_first_item = np.array(
            [0.3, 0.5,], dtype="float32"
        )
        self.assertEqual(
            resource_list[1].get_tensor_rep().tolist(),
            expected_first_item.tolist()
        )

        expected_third_item = np.array(
            [0.1, 0.4,], dtype="float32"
        )
        self.assertEqual(
            resource_list[2].get_tensor_rep().tolist(),
            expected_third_item.tolist()
        )

        expected_fourth_item = np.array(
            [0.9, 0.4,], dtype="float32"
        )
        self.assertEqual(
            resource_list[3].get_tensor_rep().tolist(),
            expected_fourth_item.tolist()
        )
    
    def test_solver(self):
        self.solver.solve(self.dummy_state)


        actual_reward,\
        actual_empty_nodes,\
        actual_num_rejected_items,\
        actual_rejected_value = compute_stats(self.solver.solution)
        
        precision = 2
        self.assertAlmostEqual(
            actual_reward[0] + actual_rejected_value[0], 1.4, precision
        )

