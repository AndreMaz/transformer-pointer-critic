import numpy as np
from environment.custom.resource_v3.node import Node
from environment.custom.resource_v3.resource import Resource
import tensorflow as tf

def reshape_into_horizontal_format(data, batch_size, decoding_steps):
    reshaped = tf.reshape(data, [decoding_steps, batch_size])
    reshaped = tf.transpose(reshaped, [1, 0])

    return reshaped

def reshape_into_vertical_format(data, batch_size):
    reshaped = tf.transpose(data, [1, 0])
    reshaped = tf.reshape(reshaped, [batch_size, 1])

    return reshaped

def round_half_up(n, decimals=0):
    multiplier = 10 ** decimals
    # Replace math.floor with np.floor
    return np.floor(n*multiplier + 0.5) / multiplier

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

def compute_remaining_resources(nodes, reqs):
    
    remaining_resources = nodes - reqs

    return remaining_resources

def compute_max_steps(nets, heuristic_solvers):
    concat_history = nets

    for solver in heuristic_solvers:
        concat_history = concat_history + solver.solution

    steps_list = []
    for node in concat_history:
        steps_list.append(len(node.CPU_history))

    return max(steps_list)


def export_to_csv(history, max_steps, method: str, location) -> None:
        with open(f"{location}_{method}.csv", 'w') as fp:

            header = f'Method;Step;Node;CPU;RAM;MEM;Delta\n'
            fp.write(header)
            for history_instance in history:
                node: Node

                delta, rejected = compute_delta(history_instance)

                for node in history_instance:
                    for step in range(max_steps):

                        try:
                            current_CPU = node.CPU_history[step]
                            current_RAM = node.RAM_history[step]
                            current_MEM = node.MEM_history[step]
                            
                        except:
                            last_step_in_node = len(node.CPU_history) - 1 
                            current_CPU = node.CPU_history[last_step_in_node]
                            current_RAM = node.RAM_history[last_step_in_node]
                            current_MEM = node.MEM_history[last_step_in_node]

                        CPU_load = current_CPU
                        RAM_load = current_RAM
                        MEM_load = current_MEM
                        
                        node_info = f'{method};{step};{node.id};{CPU_load[0]:.2f};{RAM_load[0]:.2f};{MEM_load[0]:.2f};{delta[0]:.2f}\n'
                        # print(node_info)
                        fp.write(node_info)

        fp.close()

def compute_delta(node_list) -> float:
    nodes_cpu_stats = []
    nodes_ram_stats = []
    nodes_mem_stats = []
    
    node: Node = node_list[0]
    num_rejected = len(node.req_list)

    node: Node
    for node in node_list[1:]:
        nodes_cpu_stats.append(node.remaining_CPU)
        nodes_ram_stats.append(node.remaining_RAM)
        nodes_mem_stats.append(node.remaining_MEM)

    min_cpu = min(nodes_cpu_stats)
    min_ram = min(nodes_ram_stats)
    min_mem = min(nodes_mem_stats)

    delta = min([min_cpu, min_ram, min_mem])

    return delta, num_rejected

def num_overloaded_nodes(node_list) -> int:
    num_nodes = 0

    node: Node
    for node in node_list:
        min_resource = min([
            node.remaining_CPU,
            node.remaining_RAM,
            node.remaining_MEM
        ])

        if min_resource < 0:
            num_nodes += 1

    return num_nodes