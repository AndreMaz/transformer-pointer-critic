
import sys
sys.path.append('.')

import numpy as np
from ortools.linear_solver import pywraplp

import json

from typing import List, Tuple

from environment.custom.knapsack_v2.misc.utils import round_half_up
from environment.custom.knapsack_v2.heuristic.base_heuristic import BaseHeuristic
from environment.custom.knapsack_v2.bin import Bin
from environment.custom.knapsack_v2.item import Item
from operator import itemgetter, attrgetter

class ORTools(BaseHeuristic):
    def __init__(self,
                num_nodes: int,
                opts: dict
                ):
        super(ORTools, self).__init__(num_nodes)

        self.time_limit_ms: int = opts['time_limit_ms']
        self.num_threads: int = opts['num_threads']

        self.solver = pywraplp.Solver('multiple_knapsack_mip',
                             pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
        
        # Time limit info: https://developers.google.com/optimization/lp/glop#setting-time-limits
        self.solver.SetTimeLimit(self.time_limit_ms)
        self.solver.SetNumThreads(self.num_threads)


        self.generate_name()
    
    def generate_name(self):
        self.name = f'OR_Tools_time_limit_{self.time_limit_ms}'

    def solve(self, state):
        
        bin_list = self.parse_bins(state)
        EOS_NODE = bin_list.pop(0)
        item_list = self.parse_items(state)


        data = self.parse_data(state)
        # Variables
        # x[i, j] = 1 if item i is packed in bin j.
        x = {}
        for i in data['items']:
            for j in data['bins']:
                x[(i, j)] = self.solver.IntVar(0, 1, 'x_%i_%i' % (i, j))

        # Constraints
        # Each item can be in at most one bin.
        for i in data['items']:
            self.solver.Add(sum(x[i, j] for j in data['bins']) <= 1)
        # The amount packed in each bin cannot exceed its capacity.
        for j in data['bins']:
            self.solver.Add(
                sum(x[(i, j)] * data['weights'][i]
                    for i in data['items']) <= data['bin_capacities'][j])

        # Objective
        objective = self.solver.Objective()

        for i in data['items']:
            for j in data['bins']:
                objective.SetCoefficient(x[(i, j)], data['values'][i])
        objective.SetMaximization()

        # Solve the problem
        status = self.solver.Solve()

        taken_items = np.zeros([len(data['items'])], dtype='int8')
        for entry in x:
            if x[entry].solution_value() > 0:
                # Mark item as taken
                taken_items[entry[0]] = 1

                # Insert item into the bin
                bin_list[entry[1]].insert_item(
                    item_list[entry[0]]
                )

        for index, taken in enumerate(taken_items):
            if taken == 0:
                EOS_NODE.insert_item(
                    item_list[index]
                )

        # Store a reference with the solution
        self.solution = [EOS_NODE] + bin_list

        # if status == pywraplp.Solver.OPTIMAL:
        #     print('Total packed value:', objective.Value())
            
        #     total_weight = 0
        #     for j in data['bins']:
        #         bin_weight = 0
        #         bin_value = 0
        #         print('Bin ', j, '\n')
        #         for i in data['items']:
        #             if x[i, j].solution_value() > 0:
        #                 print('Item', i, '- weight:', data['weights'][i], ' value:',
        #                         data['values'][i])
        #                 bin_weight += data['weights'][i]
        #                 bin_value += data['values'][i]
        #         print('Packed bin weight:', bin_weight)
        #         print('Packed bin value:', bin_value)
        #         print()
        #         total_weight += bin_weight
        #     print('Total packed weight:', total_weight)
        # else:
        #     print('The problem does not have an optimal solution.')
    
        return objective.Value()
       
    def parse_data(self, state):
        # state = round_half_up(state * 100, 0)
        # state[:, 0, :] = -2
        # state = state.astype('int8')
        

        # split the state into bins and items
        bins = state[0, :self.num_bins, :]
        items = state[0, self.num_bins:, :]

        # Create OR tools acceptabe format
        data = {}
        weights = []
        values = []

        for item in items:
            weights.append(item[0])
            values.append(float(item[1]))

        data['items'] = list(range(len(weights)))
        data['num_items'] = len(weights)
        data['weights'] = weights
        data['values'] = values

        bin_capacities = []
        for bin in bins[1:]: #Skip EOS bin
            bin_capacities.append(bin[0])

        data['bins'] = list(range((len(bin_capacities))))
        data['bin_capacities'] = bin_capacities

        return data

    
if  __name__ == "__main__": # pragma: no cover
    with open(f"configs/KnapsackV2.json") as json_file:
        params = json.load(json_file)

    heuristic_opts = params['tester_config']['heuristic']['or_tools']

    dummy_state = np.array([
        [
            [-2, -2],
            [ 0.1,  0.0],
            [ 0.5,  0.0],
            [ 0.2,  0.1],
            [ 0.3,  0.5],
            [ 0.1,  0.4],
        ]
    ], dtype='float32')
    
    node_sample_size = 3
    
    solver = ORTools(node_sample_size, heuristic_opts)

    solver.solve(dummy_state)

    solver.solution