import sys
sys.path.append('.')

import json
import numpy as np
import cplex

from docplex.mp.model import Model

from environment.custom.resource_v3.heuristic.base_heuristic import BaseHeuristic
from environment.custom.resource_v3.misc.utils import round_half_up

class CPLEXGreedyCritical(BaseHeuristic):
    def __init__(self,
                num_nodes: int,
                normalization_factor: int,
                opts: dict
                ):
        super(CPLEXGreedyCritical, self).__init__(num_nodes, normalization_factor)

        self.time_limit_ms: int = opts['time_limit_ms']
        self.num_threads: int = opts['num_threads']

        self.greedy_with_critical_resource: bool = opts['greedy_with_critical_resource']

        self.generate_name()

    def generate_name(self):
        self.name = f'CPLEX'

    def solve(self, state):
                
        node_list = self.parse_nodes(state)
        EOS_NODE = node_list.pop(0)

        resource_list = self.parse_resources(state)

        # CPU, RAM and MEM indexes
        feature_ids = range(state.shape[-1])

        # Rescale nodes for CPLEX to avoid precision issues
        # More info: https://github.com/IBMDecisionOptimization/docplex-examples/issues/46
        nodes = round_half_up(
            state[:, 1:self.num_nodes][0] * self.normalization_factor,
            0 # Decimal precision
        )
        nodes_ids = range(nodes.shape[0])

        # Rescale resources for CPLEX to avoid precision issues
        # More info: https://github.com/IBMDecisionOptimization/docplex-examples/issues/46
        resources = round_half_up(
            state[:, self.num_nodes:][0] * self.normalization_factor,
            0 # Decimal precision
        )
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
        # Rescale Upper Bound to avoid precision issues
        # More info: https://github.com/IBMDecisionOptimization/docplex-examples/issues/46
        omega_node = mdl.continuous_var_list([n for n in nodes_ids], lb=0, ub=1*self.normalization_factor, name='omega')
        
        # Global critical resource
        # Rescale Upper Bound to avoid precision issues
        # More info: https://github.com/IBMDecisionOptimization/docplex-examples/issues/46
        omega_max = mdl.continuous_var(name='omega_max', lb=0, ub=1 * self.normalization_factor)
        mdl._is_continuous_var(omega_max)
        #mdl.add_constraint(omega_max >= 0)
        #mdl.add_constraint(omega_max <= 1)


        # Rule placement
        mdl.add_constraints(
            mdl.sum(is_resource_placed_at_node[x, n] for n in nodes_ids ) == is_resource_executed[x] for x in resources_ids
        )

        # Computation Resource Limitation
        mdl.add_constraints(
            mdl.sum(float(resources[x, m]) * is_resource_placed_at_node[x, n] for x in resources_ids) <= float(nodes[n, m]) for n in nodes_ids for m in feature_ids
        )

        # Determining the Critical Resource Margin
        mdl.add_constraints(
            omega_node[n] <= float(nodes[n, m]) - mdl.sum(float(resources[x, m]) * is_resource_placed_at_node[x, n] for x in resources_ids) for n in nodes_ids for m in feature_ids
        )

        # Most critical Resource
        mdl.add_constraints(
            omega_max <= omega_node[n] for n in nodes_ids
        )

        # Objective function
        if self.greedy_with_critical_resource:
             # Greedy + Most Critical Resource
            mdl.maximize(mdl.sum(is_resource_executed[x] for x in resources_ids) + omega_max / self.normalization_factor)
        else:
            # Greedy
            mdl.maximize(mdl.sum(is_resource_executed[x] for x in resources_ids))
        # mdl.maximize(mdl.sum(is_resource_executed[x] for x in resources_ids)) # Greedy
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

        # Store the status code of the solution
        # More info about CPLEX status codes here: https://www.ibm.com/docs/en/icos/12.8.0.0?topic=micclcarm-solution-status-codes-by-number-in-cplex-callable-library-c-api
        if mdl.solution.solve_details.status_code == 101: # 101 is MIP Optimal
            self.is_optimal = 1

        # else:
        #    self.is_optimal = 0 # Set to 0 by default in the BaseHeuristic class

        # mdl.export_as_lp('a.lp')
    
if  __name__ == "__main__": # pragma: no cover
    with open(f"configs/ResourceV3.json") as json_file:
        params = json.load(json_file)

    heuristic_opts = params['tester_config']['heuristic']['cplex']

    ###################################
    # node_sample_size = 4
    # dummy_state = np.array([
    #     [   
    #         # Nodes
    #         # CPU   RAM  MEM
    #         [-2.0, -2.0, -2.0],
    #         [ 0.5,  0.5, 0.3],
    #         [ 0.9,  0.9, 0.9],
    #         [ 0.4,  0.8, 0.9],

    #         # Resources
    #         # CPU  RAM   MEM
    #         [0.3,  0.3, 0.3],
    #         [0.5,  0.5, 0.5],
    #         [0.5,  0.5, 1.9],
    #     ]
    # ], dtype='float32')
    ###################################    
    
    ###################################
    node_sample_size = 3
    normalization_factor = 100
    dummy_state = np.array([
            [
                # Nodes
                # CPU   RAM  MEM
                [-2.0, -2.0, -2.0],
                [0.5,  1.1,  0.9], # Should Place both reqs here
                [0.1,  0.1,  0.1],

                # Resources
                # CPU  RAM   MEM
                [0.2,  0.5,  0.4], 
                [0.3,  0.5,  0.4], 
            ]
        ], dtype='float32')

    solver = CPLEXGreedyCritical(
        node_sample_size, normalization_factor, heuristic_opts
    )

    solver.solve(dummy_state)

    solver.solution
