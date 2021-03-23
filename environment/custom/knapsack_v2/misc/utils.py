from typing import List
import tensorflow as tf
import numpy as np

def round_half_up(n, decimals=0):
    multiplier = 10 ** decimals
    # Replace math.floor with np.floor
    return np.floor(n*multiplier + 0.5) / multiplier

def compute_remaining_resources(bins, items, decimal_precision = 2):
    
    # bins_current_load = bins[:, 1]
    # items_weight = items[:, 0]
    updated_bin = bins.copy()
    
    updated_bin[:, 1] = updated_bin[:, 1] + items[:, 0]

    updated_bin = round_half_up(updated_bin, decimal_precision)

    return updated_bin

def bins_eos_checker(bins, EOS_SYMBOL, num_features, result_dtype = 'int32'):
    # Check if selected bins are EOS
    # Marked as 1 = EOS node
    # Marked as 0 = not a EOS node
    is_eos_bin = tf.cast(tf.equal(bins, EOS_SYMBOL), dtype = result_dtype)
    # if all elements are equal to EOS code
    # the result should be equal to the number of features
    is_eos_bin = tf.reduce_sum(is_eos_bin, axis=-1)
    is_eos_bin = tf.cast(tf.equal(is_eos_bin, num_features), dtype = result_dtype)

    return is_eos_bin

def compute_max_steps(nets, heuristic_solvers):
    concat_history = nets

    for solver in heuristic_solvers:
        concat_history = concat_history + solver.solution

    steps_list = []
    for node in concat_history:
        steps_list.append(len(node.load_history))

    return max(steps_list)

def compute_stats(node_list) -> float:
    # Find number of rejected requests
    eos_node = node_list[0]
    num_rejected = len(eos_node.item_list)

    # Find dominant resource AND
    # Find the number of unused nodes
    nodes_cpu_stats = []
    nodes_ram_stats = []
    nodes_mem_stats = []
    
    reward = 0
    empty_nodes = 0

    for node in node_list[1:]:
        nodes_cpu_stats.append(node.remaining_CPU)
        nodes_ram_stats.append(node.remaining_RAM)
        nodes_mem_stats.append(node.remaining_MEM)

        if len(node.item_list) == 0:
            empty_nodes += 1

    min_cpu = min(nodes_cpu_stats)
    min_ram = min(nodes_ram_stats)
    min_mem = min(nodes_mem_stats)

    delta = min([min_cpu, min_ram, min_mem])

    return delta, num_rejected, empty_nodes


def gather_stats_from_solutions(env, heuristic_solvers) -> List[dict]:
    stats = []
    
    won = 0
    lost = 0
    draw = 0

    net_dominant, net_rejected, empty_nodes = compute_stats(env.history[0])
    net_dominant = round_half_up(net_dominant, 2)

    stats.append({
            'net_dominant': net_dominant,
            'net_rejected': net_rejected,
            'net_empty_nodes': empty_nodes
    })
    
    max_dominant = -1
    min_rejected = 9999

    for solver in heuristic_solvers:
        solver_dominant, solver_rejected, solver_empty_nodes = compute_stats(solver.solution)

        max_dominant = max(max_dominant, round_half_up(solver_dominant, 2))
        min_rejected = min(min_rejected, solver_rejected)

        stats.append({
            f'{solver.name}_dominant': solver_dominant,
            f'{solver.name}_rejected': solver_rejected,
            f'{solver.name}_empty_nodes': solver_empty_nodes,
        })

    dominant_result = np.array([
        0, # Won
        0, # Draw
        0, # Loss
    ])

    if max_dominant > net_dominant:
        dominant_result[2] = 1 # Lost
    elif max_dominant == net_dominant:
        dominant_result[1] = 1 
    else: 
        dominant_result[0] = 1

    rejected_result = np.array([
        0, # Won
        0, # Draw
        0, # Loss
    ])

    if min_rejected < net_rejected:
        rejected_result[2] = 1 # Lost
    elif min_rejected == net_rejected:
        rejected_result[1]  = 1
    else:
        rejected_result[0] = 1

    return stats, dominant_result, rejected_result

def generate_file_name(agent_config):
    gamma = agent_config['gamma']
    entropy = agent_config['entropy_coefficient']
    num_layers = agent_config['actor']['num_layers']
    
    actor_lr = agent_config['actor']['learning_rate']
    critic_lr = agent_config['critic']['learning_rate']

    file_name = f"num_l:{num_layers}|e:{entropy}|ac_lr:{actor_lr}|cr_lr:{critic_lr}"

    return file_name
