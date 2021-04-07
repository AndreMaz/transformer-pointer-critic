import sys
sys.path.append('.')

import json
import numpy as np
import cplex

from docplex.mp.model import Model

from environment.custom.resource_v3.heuristic.base_heuristic import BaseHeuristic
from environment.custom.resource_v3.misc.utils import round_half_up

class CPLEXSolver(BaseHeuristic):
    def __init__(self,
                num_nodes: int,
                opts: dict
                ):
        super(CPLEXSolver, self).__init__(num_nodes)

        self.time_limit_ms: int = opts['time_limit_ms']
        self.num_threads: int = opts['num_threads']

    def generate_name(self, state):
        self.name = f'CPLEX'

    def solve(self, state):
                
        node_list = self.parse_nodes(state)
        EOS_NODE = node_list.pop(0)

        resource_list = self.parse_resources(state)

        # CPU, RAM and MEM indexes
        feature_ids = range(state.shape[-1])

        nodes = state[:, 1:self.num_nodes][0]
        nodes_ids = range(nodes.shape[0])

        resources = state[:, self.num_nodes:][0]
        resources_ids = range(resources.shape[0])
        
        # CPLEX Model
        mdl = Model(name="load_balancing")

        mdl.parameters.threads = self.num_threads
        # In seconds
        mdl.parameters.timelimit = int(self.time_limit_ms / 1000)

        is_resource_executed = mdl.binary_var_list(
            [x for x in resources_ids], name='w')

        is_resource_placed_at_node = mdl.binary_var_dict(
            [(x, n) for x in resources_ids for n in nodes_ids], name='B')

        # Critical Resource at specific nodes
        omega_node = mdl.continuous_var_list([n for n in nodes_ids], lb=0, ub=1, name='omega')
        #mdl.add_constraints([omega_node[n] >= 0 for n in nodes_ids])
        # mdl.add_constraints([omega_node[n] <= 1 for n in nodes_ids])

        # Global critical resource
        omega_max = mdl.continuous_var(name='omega_max', lb=0, ub=1)
        mdl._is_continuous_var(omega_max)
        #mdl.add_constraint(omega_max >= 0)
        #mdl.add_constraint(omega_max <= 1)


        # Rule placement
        mdl.add_constraints(
            mdl.sum(is_resource_placed_at_node[x, n] for n in nodes_ids ) == is_resource_executed[x] for x in resources_ids
        )

        # Critical Resource
        mdl.add_constraints(
            omega_node[n] >= mdl.sum(float(resources[x, m]) * is_resource_placed_at_node[x, n] for x in resources_ids) for n in nodes_ids for m in feature_ids
        )

        # Most critical Resource
        mdl.add_constraints(
            omega_max >= omega_node[n] for n in nodes_ids
        )

        # Computation Resource Limitation
        mdl.add_constraints(
            mdl.sum(float(resources[x, m]) * is_resource_placed_at_node[x, n] for x in resources_ids) <= float(nodes[n, m]) for n in nodes_ids for m in feature_ids
        )

        # Objective function
        mdl.maximize(mdl.sum(is_resource_executed[x] for x in resources_ids) - omega_max)
        # mdl.print_information()
        mdl.solve()
        # mdl.print_solution(print_zeros=True)
        
        # Parse solution
        placements = mdl.solution.get_value_dict(is_resource_placed_at_node)
        taken_items = np.zeros([len(resources_ids)], dtype='int8')

        for entry in placements:
            if placements[entry] == 1:
                # Mark as taken
                taken_items[entry[0]] = 1

                # Insert into the node
                node_list[entry[1]].insert_req(
                    resource_list[entry[0]]
                )
        
        # Place the rejected into EOS
        for index, taken in enumerate(taken_items):
            if taken == 0:
                EOS_NODE.insert_req(
                    resource_list[index]
                )

        # Store a reference with the solution
        self.solution = [EOS_NODE] + node_list

        # mdl.export_as_lp('a.lp')
    
if  __name__ == "__main__": # pragma: no cover
    with open(f"configs/KnapsackV2.json") as json_file:
        params = json.load(json_file)

    heuristic_opts = params['tester_config']['heuristic']['or_tools']

    dummy_state = np.array([
        [   
            # Nodes
            # CPU   RAM  MEM
            [-2.0, -2.0, -2.0],
            [ 0.5,  0.5, 0.3],
            [ 0.9,  0.9, 0.9],
            [ 0.4,  0.8, 0.9],

            # Resources
            # CPU  RAM   MEM
            [0.3,  0.3, 0.3],
            [0.5,  0.5, 0.5],
            [0.5,  0.5, 1.9],
        ]
    ], dtype='float32')
    
    node_sample_size = 4
    
    solver = CPLEXSolver(node_sample_size, heuristic_opts)

    solver.solve(dummy_state)

    solver.solution