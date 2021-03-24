from environment.custom.knapsack_v2.heuristic.random_heuristic import RandomHeuristic
from environment.custom.knapsack_v2.heuristic.waste_reduction import WasteReductionHeuristic
from environment.custom.knapsack_v2.heuristic.or_tools import ORTools

def heuristic_factory(num_nodes: int, opts: dict):
    heuristic_list = []
    
    waste_reduction_solvers = generate_waste_reduction_combos(
        num_nodes,
        opts['waste_reduction']
    )

    random_solvers = [
        RandomHeuristic(num_nodes, opts['random'])
    ]

    or_tools = [
        ORTools(num_nodes, opts['or_tools'])
    ]

    # Concat the array with the solvers
    heuristic_list = heuristic_list + waste_reduction_solvers + random_solvers + or_tools

    return heuristic_list

def generate_waste_reduction_combos(num_nodes: int, opts: dict):
    generate_combos: bool = opts['generate_params_combos']

    # Return as is
    if not generate_combos:
        return [
                WasteReductionHeuristic(num_nodes, opts)
        ]

    dominant_list = []
    
    item_sorting = [True, False]
    bin_sorting = [True, False]
    
    for i in item_sorting:
        for b in bin_sorting:
            opts_combo = {
              "item_sort_descending": i,
              "bin_sort_descending": b
            }

            dominant_list.append(
                WasteReductionHeuristic(num_nodes, opts_combo)
            )

    return dominant_list