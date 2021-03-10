# For plotting
# from environment.custom.knapsack.heuristic import solver
import matplotlib.pyplot as plt
import os
import numpy as np
from environment.custom.resource_v3.misc.utils import generate_file_name
from environment.custom.resource_v3.misc.csv_writer import log_training_stats

def plotter(data, env, agent, agent_config: dict, trainer_config: dict, log_dir: str):
    
    export_stats: bool = trainer_config['export_stats']['export_stats']
    folder: bool = trainer_config['export_stats']['folder']

    if not export_stats: 
        return

    file_name = generate_file_name(agent_config)
    location = os.path.join(log_dir, folder)

    plotter_leaning(data, location, 'learning', agent.name)

    plotter_rewards(data, location, 'rewards', agent.name)
    
    log_training_stats(data, location, 'logs')

def plotter_leaning(data, location, file_name, agent_name ):
    # Destructure the tuple
    _,\
        _,\
        _,\
        value_loss_buffer, \
        bins_policy_loss_buffer,\
        bins_total_loss_buffer,\
        bins_entropy_buffer = data
    
    x_values = [i for i in range(len(value_loss_buffer))]

    plt.plot(x_values, value_loss_buffer, label="Value Loss")
    
    plt.plot(x_values, bins_policy_loss_buffer, label="Bin Policy Net Loss")
    plt.plot(x_values, bins_total_loss_buffer, label="Total Bin Net Loss")
    plt.plot(x_values, bins_entropy_buffer, label="Bin Net Entropy")

    plt.xlabel('Episode')

    plot_title = f"{agent_name.upper()}\n|" + file_name
    plt.title(plot_title)

    # Check if dir exists. If not, create it
    if not os.path.isdir(location):
        os.makedirs(location)

    # Show legend info
    plt.legend()

    # plt.show(block=blockPlot)
    plt.savefig(
        f"{location}/{file_name.replace(' ', '')}.png",
        dpi = 200,
        bbox_inches = "tight"
    )

    plt.close()
    
def plotter_rewards(data, location, file_name, agent_name ):
    
    # Destructure the tuple
    average_rewards_buffer,\
        min_rewards_buffer,\
        max_rewards_buffer,\
        _, \
        _,\
        _,\
        _, = data

    x_values = [i for i in range(len(average_rewards_buffer))]
    
    plt.plot(x_values, average_rewards_buffer, label="Average (in batch) Double Pointer Critic")
    plt.plot(x_values, min_rewards_buffer, label="Minimum (in Batch) Double Pointer Critic")
    plt.plot(x_values, max_rewards_buffer, label="Maximum (in batch) Double Pointer Critic")

    plt.ylabel('Collected Reward')
    plt.xlabel('Episode')

    plot_title = f"{agent_name.upper()}\n|" + file_name
    plt.title(plot_title)

    # Check if dir exists. If not, create it
    if not os.path.isdir(location):
        os.makedirs(location)

    # Show legend info
    plt.legend()

    # plt.show(block=blockPlot)
    plt.savefig(
        f"{location}/{file_name.replace(' ', '')}.png",
        dpi = 200,
        bbox_inches = "tight"
    )

    plt.close()

def plot_attentions(attentions,
                    num_resources,
                    num_bins
                ):
    
    fig, axs = plt.subplots(num_resources, 2)

    for index, attention in enumerate(attentions):
        
        # Only show the attention over the resources
        resource_attention = attention['resource_attention'][:, num_bins:]
        axs[index, 0].matshow(np.transpose(resource_attention))
        # axs[index, 0].set_title('Item Attention')

        # Only show the attention over the bins
        bin_attention = attention['bin_attention'][:, :num_bins]
        axs[index, 1].matshow(np.transpose(bin_attention))
        # axs[index, 1].set_title('Backpack Attention')

    for index in range(num_resources):
        # Select the plot by index for the Items
        plt.sca(axs[index, 0])
        # Add the ticks and the labels
        resource_input = attentions[index]["resource_net_input"]

        CPU = int(round(resource_input[0,0,0]  * 100))
        RAM = int(round(resource_input[0,0,1]  * 100))
        MEM = int(round(resource_input[0,0,2]  * 100))

        resource_xlabel = f'C:{CPU} R:{RAM} M:{MEM}'
        plt.xticks([0], [resource_xlabel], fontsize=8)

        resource_states = attentions[index]['current_state'][0, num_bins:]
        resource_ylabel = []
        for itm in resource_states:
            CPU = int(round(itm[0] * 100))
            RAM = int(round(itm[1] * 100))
            MEM = int(round(itm[2] * 100))


            resource_ylabel.append(
                f'C:{CPU} R:{RAM} M:{MEM}'
            )

        plt.yticks(range(len(resource_ylabel)), resource_ylabel, rotation=0, fontsize=8)

        # Select the plot by index for the Backpacks
        plt.sca(axs[index, 1])
        # Add the ticks and the labels
        resource_input = attentions[index]["bin_net_input"]
        CPU = int(round(resource_input[0,0,0] * 100))
        RAM = int(round(resource_input[0,0,1] * 100))
        MEM = int(round(resource_input[0,0,2] * 100))

        bin_xlabel = f'C:{CPU} R:{RAM} M:{MEM}'
        plt.xticks([0], [bin_xlabel], fontsize=8)

        bin_states = attentions[index]['current_state'][0, :num_bins]
        bin_ylabel = []
        for bp in bin_states:
            CPU = int(round(bp[0] * 100)  )
            RAM = int(round(bp[1] * 100)  )
            MEM = int(round(bp[2] * 100)  )

            bin_ylabel.append(
                f'C:{CPU} R:{RAM} M:{MEM}'
            )
        plt.yticks(range(len(bin_ylabel)), bin_ylabel, rotation=0, fontsize=8)
    
    # plt.subplots_adjust(wspace=0.3, hspace = 0.3)
    plt.tight_layout()
    plt.show(block=True)
