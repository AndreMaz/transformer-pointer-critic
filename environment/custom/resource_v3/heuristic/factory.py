import sys
sys.path.append('.')
import json
from environment.custom.resource_v3.heuristic.dominant_heuristic import DominantResourceHeuristic
from environment.custom.resource_v3.heuristic.random_heuristic import RandomHeuristic


def heuristic_factory(num_nodes: int, opts: dict):
    heuristic_list = []
    
    dominant_solvers = generate_dominant_combos(num_nodes,
                                            opts['dominant_resource']
                                        )

    random_solvers = [
        RandomHeuristic(num_nodes, opts['random'])
    ]
    
    # Concat the array with the solvers
    heuristic_list = heuristic_list + dominant_solvers + random_solvers

    return heuristic_list

def generate_dominant_combos(num_nodes: int, opts: dict):
    generate_combos: bool = opts['generate_params_combos']

    # Return as is
    if not generate_combos:
        return [
                DominantResourceHeuristic(num_nodes, opts)
        ]

    dominant_list = []
    
    resource_sorting = [True, False]
    node_sorting = [True, False]
    
    for r in resource_sorting:
        for n in node_sorting:
            opts_combo = {
              "resource_sort_descending": r,
              "node_sort_descending": n
            }

            dominant_list.append(
                DominantResourceHeuristic(num_nodes, opts_combo)
            )

    return dominant_list


if __name__ == "__main__":
    env_name = 'ResourceEnvironmentV3'
    
    with open(f"configs/ResourceV3.json") as json_file:
        params = json.load(json_file)

    tester_configs = params['tester_config']
    num_nodes = 2

    heuristic_factory(num_nodes, tester_configs['heuristic'])