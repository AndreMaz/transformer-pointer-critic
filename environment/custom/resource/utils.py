import numpy as np


def compute_max_steps(nets, heuristic):
    concat_history = nets + heuristic

    steps_list = []
    for node in concat_history:
        steps_list.append(len(node.CPU_history))

    return max(steps_list)

def export_to_csv(history, max_steps, method: str, location) -> None:
        with open(location, 'w') as fp:

            header = f'Method;Step;Node;CPU;RAM;MEM;Percentage_Penalized\n'
            fp.write(header)
            for history_instance in history:
                # node: History
                steps_list = []
                for node in history_instance:
                    steps_list.append(len(node.CPU_history))

                for node in history_instance:
                    # max_steps = max(steps_list)
                    for step in range(max_steps):
                        try:
                            current_CPU = node.CPU_history[step]
                            current_RAM = node.RAM_history[step]
                            current_MEM = node.MEM_history[step]
                            percentage_penalized = node.percentage_penalized_history[step]
                        except:
                            last_step_in_node = len(node.CPU_history) - 1 
                            current_CPU = node.CPU_history[last_step_in_node]
                            current_RAM = node.RAM_history[last_step_in_node]
                            current_MEM = node.MEM_history[last_step_in_node]
                            percentage_penalized = node.percentage_penalized_history[last_step_in_node]

                        CPU_load = np.array([0], dtype='float32')
                        RAM_load = np.array([0], dtype='float32')
                        MEM_load = np.array([0], dtype='float32')
                        
                        # Don't compute for EOS node
                        if node.id != 0 and len(node.resources) != 0:
                            CPU_load = (1 - current_CPU / node.CPU) * 100
                            RAM_load = (1 - current_RAM / node.RAM) * 100
                            MEM_load = (1 - current_MEM / node.MEM) * 100

                        node_info = f'{method};{step};{node.id};{CPU_load[0]:.2f};{RAM_load[0]:.2f};{MEM_load[0]:.2f};{percentage_penalized:.2f}\n'
                        # print(node_info)
                        fp.write(node_info)

        fp.close()