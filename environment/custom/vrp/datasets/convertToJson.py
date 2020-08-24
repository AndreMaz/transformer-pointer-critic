import json
from os import listdir, write
from os.path import isfile, join
NODE = 'node'
DEPOT = 'depot'

def converter(dir_path):
    prob_sol_pair = find_problems_with_opt(dir_path)

    dataset = []
    for prob_name, sol_name in prob_sol_pair:
        p = parse_problem(dir_path, prob_name)
        s = parse_solution(dir_path, sol_name)
        entry = {
            'problem': p,
            'solution': s
        }
        # dataset.append(entry)
        with open(f'{dir_path}/{prob_name}.json', 'w') as fp:
            json.dump(entry, fp, indent=4)

    return dataset

def parse_solution(dir_path, sol_name):
    file = open(f'{dir_path}/{sol_name}', 'r')
    
    cost = - 1
    route_list = []
    for line in file:
        line = line.split()
        if len(line) == 0:
            break
        if line[0] == "Route":
            route = []
            print(line[2:])
            for node_id in line[2:]:
                route.append(int(node_id)) 
            route_list.append(route)
        if line[0] == "cost":
            cost = int(line[-1])

    solution = {
        "cost": cost,
        "num_routes": len(route_list),
        "routes": route_list,
    }
    return solution

def parse_problem(dir_path, prob_name):
    file = open(f'{dir_path}/{prob_name}', 'r')
    # line_num = 0
    NODE_LOC = False
    DEMAND_SEC = False
    DEPOT_SEC = False

    name = ''
    comment = ' '
    prob_type = ''
    dimension = - 1
    edge_weight_type = ''
    nodes = []
    vehicle_capacity = -1 # Will be filled during parsing
    
    for line in file:
        line = line.split()
        if line[-1] == 'EOF':
            break
        if line[0] == 'NAME':
            name = line[-1]
        if line[0] == 'COMMENT':
            comment = comment.join(line[2:])
            continue
        if line[0] == 'TYPE':
            prob_type = line[-1]
        if line[0] == 'DIMENSION':
            dimension = int(line[-1])
            continue
        if line[0] == 'EDGE_WEIGHT_TYPE':
            edge_weight_type = line[-1]
            continue
        if line[0] == 'CAPACITY':
            # print(line[-1])
            vehicle_capacity = int(line[-1])
            continue
        if line[-1] == 'NODE_COORD_SECTION':
            NODE_LOC = True
            DEMAND_SEC = False
            DEPOT_SEC = False
            continue
        if line[-1] == 'DEMAND_SECTION':
            NODE_LOC = False
            DEMAND_SEC = True
            DEPOT_SEC = False
            continue
        if line[-1] == 'DEPOT_SECTION':
            NODE_LOC = False
            DEMAND_SEC = False
            DEPOT_SEC = True
            continue
            
        if NODE_LOC:
            node = {
                "id": int(line[0]),
                "x": int(line[1]),
                "y": int(line[2]),
                "demand": -1, # Will be filled later
                'type': NODE
            }
            nodes.append(node)
        
        if DEMAND_SEC:
            id = int(line[0])
            node = nodes[id - 1] # In the datasets the indexes start at 1 in
            node['demand'] = int(line[1])
        
        if DEPOT_SEC:
            if (line[0] != -1): # -1 is the EOL of depot section
                id = int(line[0])
                node = nodes[id - 1] # In the datasets the indexes start at 1 in
                node['type'] = DEPOT

        # line_num += 1
    problem = {
        "name": name,
        "comment": comment,
        "type": prob_type,
        "dimension": dimension,
        "edge_weight_type": edge_weight_type,
        "capacity": vehicle_capacity,
        "nodes": nodes,
    }

    return problem

def find_problems_with_opt(dir_path):
    prob_sol_pair = []
    file_list = listdir(dir_path)
    for file_name in file_list:
        file_type = file_name.split(".")
        if file_type[-1] == 'vrp':
            try:
                opt_index = file_list.index(f'opt-{file_type[0]}')
                p_name = file_name
                sol_name = file_list[opt_index]

                prob_sol_pair.append(
                    (p_name, sol_name)
                )
            except ValueError as identifier:
                # print(identifier)
                pass
    
    return prob_sol_pair

if __name__ == "__main__":
    # converter(dir_path='.')
    converter(dir_path='./environment/custom/vrp/datasets')