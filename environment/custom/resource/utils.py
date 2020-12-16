import numpy as np
from environment.custom.resource.node import Node
from environment.custom.resource.resource import Resource
import tensorflow as tf

def is_premium_wrongly_rejected_checker(are_bins_full, users_type, is_eos_bin):
    # If placed PREMIUM REQUEST at EOS while there were available nodes
    is_premium_wrongly_rejected = tf.cast(tf.logical_and(
        # Checks if is EOS bin but there is still available nodes
        tf.greater(is_eos_bin, are_bins_full),
        tf.equal(users_type, 1),
    ), dtype='int32')

    return is_premium_wrongly_rejected

def bins_eos_checker(bins, EOS_SYMBOL, num_features):
    # Check if selected bins are EOS
    # Marked as 1 = EOS node
    # Marked as 0 = not a EOS node
    is_eos_bin = tf.cast(tf.equal(bins, EOS_SYMBOL), dtype="int32")
    # if all elements are equal to EOS code
    # the result should be equal to the number of features
    is_eos_bin = tf.reduce_sum(is_eos_bin, axis=-1)
    is_eos_bin = tf.cast(tf.equal(is_eos_bin, num_features), dtype="int32")

    return is_eos_bin
    

def bins_full_checker(feasible_mask, num_features):
        # Check if all nodes are full
        # if all nodes are full then results should be equal to num_features - 1 
        # num_feature - 1 because EOS is allways available
        # Marked as 1 = All full
        # Marked as 0 = NOT all full
        are_bins_full = tf.reduce_sum(feasible_mask[:, 1:], axis=-1)
        are_bins_full = tf.cast(tf.equal(are_bins_full, num_features - 1), dtype="int32")

        return are_bins_full


def compute_max_steps(nets, heuristic):
    concat_history = nets + heuristic

    steps_list = []
    for node in concat_history:
        steps_list.append(len(node.CPU_history))

    return max(steps_list)

def export_to_csv(history, max_steps, method: str, location) -> None:
        with open(location, 'w') as fp:

            header = f'Method;Step;Node;CPU;RAM;MEM;Percentage_Penalized;Free;Premium;Resource;Batch\n'
            fp.write(header)
            for history_instance in history:
                node: Node
                steps_list = []
                for node in history_instance:
                    steps_list.append(len(node.CPU_history))

                for node in history_instance:
                    # max_steps = max(steps_list)
                    # node.print()
                    free_requests = 0
                    premium_request = 0
                    for step in range(max_steps):
                        try:
                            current_CPU = node.CPU_history[step]
                            current_RAM = node.RAM_history[step]
                            current_MEM = node.MEM_history[step]
                            percentage_penalized = node.percentage_penalized_history[step]

                            resource: Resource = node.resources[step]

                            resource_type = int(resource.request_type[0])
                            resource_batch = resource.batch_id
                            if resource_type == 0:
                                free_requests += 1
                            else:
                                premium_request += 1
                        except:
                            last_step_in_node = len(node.CPU_history) - 1 
                            current_CPU = node.CPU_history[last_step_in_node]
                            current_RAM = node.RAM_history[last_step_in_node]
                            current_MEM = node.MEM_history[last_step_in_node]
                            percentage_penalized = node.percentage_penalized_history[last_step_in_node]

                            resource_type = 'NaN'
                            resource_batch = 'NaN'

                        CPU_load = np.array([0], dtype='float32')
                        RAM_load = np.array([0], dtype='float32')
                        MEM_load = np.array([0], dtype='float32')
                        
                        # Don't compute for EOS node
                        if node.id != 0 and len(node.resources) != 0:
                            CPU_load = (1 - current_CPU / node.CPU) * 100
                            RAM_load = (1 - current_RAM / node.RAM) * 100
                            MEM_load = (1 - current_MEM / node.MEM) * 100

                        node_info = f'{method};{step};{node.id};{CPU_load[0]:.2f};{RAM_load[0]:.2f};{MEM_load[0]:.2f};{percentage_penalized:.2f};{free_requests};{premium_request};{resource_type};{resource_batch}\n'
                        # print(node_info)
                        fp.write(node_info)

        fp.close()