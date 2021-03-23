from environment.custom.knapsack_v2.heuristic.random_heuristic import RandomHeuristic

def heuristic_factory(num_nodes: int, opts: dict):
    heuristic_list = []
    
    dominant_solvers = []

    random_solvers = [
        RandomHeuristic(num_nodes, opts['random'])
    ]

    # Concat the array with the solvers
    heuristic_list = heuristic_list + dominant_solvers + random_solvers

    return heuristic_list