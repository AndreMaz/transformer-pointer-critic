import numpy as np
from environment.custom.resource_v3.misc.utils import compute_stats
from typing import List

def log_testing_stats(global_stats, folder, file_name):

    with open(f"{folder}/{file_name}.csv", 'w') as fp:
        # First passage to generate the header
        header = ''

        # Place Node and Resource stats into header
        for entry in global_stats[0]:
            if entry != 'instance':
                header = header + f"{entry};"

        # Place Heuristic method name into header
        for entry in global_stats[0]['instance']:
            keys = list(entry.keys())
            header = header + f"{' '.join(keys[0].split('_'))};{' '.join(keys[1].split('_'))};{' '.join(keys[2].split('_'))};{' '.join(keys[3].split('_'))};"
        fp.write(f"{header}\n")

        
        for instance_stats in global_stats:
            data = f"{instance_stats['test_instance']};" \
                   f"{instance_stats['bin_sample_size']};" \
                   f"{instance_stats['bin_min_value']};" \
                   f"{instance_stats['bin_max_value']};" \
                   f"{instance_stats['item_sample_size']};"

            for entry in instance_stats['instance']:
                reward, empty_nodes, num_rejected_items, rejected_value = list(entry.values())
                # dominant = round_half_up(dominant, 2)
                data = data + f"{reward[0]:.3f};{empty_nodes};{num_rejected_items};{rejected_value[0]:.3f};"
            # print(data)
            fp.write(f"{data}\n")

        fp.close()

def log_training_stats(data, location, file_name):
    average_rewards_buffer,\
        min_rewards_buffer,\
        max_rewards_buffer,\
        value_loss_buffer, \
        bins_policy_loss_buffer,\
        bins_total_loss_buffer,\
        bins_entropy_buffer = data

    with open(f"{location}/{file_name}.csv", 'w') as fp:
        header = 'Step;Avg Reward;Max Reward;Min Reward;Value Loss;Bin Entropy;Total Bin Loss;Bin Policy Loss'

        fp.write(f'{header}\n')

        for index in range(len(average_rewards_buffer)):
            avg = average_rewards_buffer[index]
            min = min_rewards_buffer[index]
            max = max_rewards_buffer[index]
            v_loss = value_loss_buffer[index]

            b_policy_loss = bins_policy_loss_buffer[index]
            b_total_loss = bins_total_loss_buffer[index]
            b_entr = bins_entropy_buffer[index]

            data = f"{index};{avg:.3f};{max:.3f};{min:.3f};{v_loss:.3f};{b_entr:.3f};{b_total_loss:.3f};{b_policy_loss:.3f}"

            fp.write(f"{data}\n")

    fp.close()

def export_to_csv(history, max_steps, method: str, location) -> None:
        with open(f"{location}_{method}.csv", 'w') as fp:

            header = f'Method;Step;Node;CPU;RAM;MEM;Delta;Rejected;Empty Nodes\n'
            fp.write(header)
            for history_instance in history:

                delta, rejected, empty_nodes = compute_stats(history_instance)

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
                        
                        node_info = f'{method};{step};{node.id};{CPU_load[0]:.2f};{RAM_load[0]:.2f};{MEM_load[0]:.2f};{delta[0]:.2f};{rejected};{empty_nodes}\n'
                        # print(node_info)
                        fp.write(node_info)

        fp.close()
