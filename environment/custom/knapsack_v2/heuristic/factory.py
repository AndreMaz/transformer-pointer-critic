def heuristic_factory(num_nodes: int, opts: dict):
    heuristic_list = []
    
    dominant_solvers = []

    random_solvers = []

    # Concat the array with the solvers
    heuristic_list = heuristic_list + dominant_solvers + random_solvers

    return heuristic_list