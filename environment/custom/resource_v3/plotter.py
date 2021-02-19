# For plotting
# from environment.custom.knapsack.heuristic import solver
import matplotlib.pyplot as plt
import os
import numpy as np
from environment.custom.resource_v3.utils import log_training_stats, generate_file_name
# Import Google OR Tools Solver
# from agents.optimum_solver import solver

STEPS = 2

def plotter(data, env, agent, agent_config, write_data_to_file = True):
    file_name = generate_file_name(agent_config) 
    location = f"./media/plots/{env.name}/{agent.name}"

    plotter_leaning(data, location, file_name+'learning', agent.name)

    plotter_rewards(data, location, file_name+'rewards', agent.name)

    if write_data_to_file:
        log_training_stats(data, location, file_name+'logs')

def plotter_leaning(data, location, file_name, agent_name ):
# Destructure the tuple
    _,\
        _,\
        _,\
        value_loss_buffer, \
        resources_policy_loss_buffer,\
        resources_total_loss_buffer,\
        resources_entropy_buffer,\
        bins_policy_loss_buffer,\
        bins_total_loss_buffer,\
        bins_entropy_buffer = data
    
    # value_loss_buffer = average_per_steps(value_loss_buffer, STEPS)
    # resources_loss_buffer = average_per_steps(resources_loss_buffer, STEPS)
    # resources_entropy_buffer = average_per_steps(resources_entropy_buffer, STEPS)
    # bins_loss_buffer = average_per_steps(bins_loss_buffer, STEPS)
    # bins_entropy_buffer = average_per_steps(bins_entropy_buffer, STEPS)

    x_values = [i for i in range(len(value_loss_buffer))]

    plt.plot(x_values, value_loss_buffer, label="Value Loss")

    plt.plot(x_values, resources_policy_loss_buffer, label="Resource Policy Net Loss")
    plt.plot(x_values, resources_total_loss_buffer, label="Total Resource Net Loss")
    plt.plot(x_values, resources_entropy_buffer, label="Resource Net Entropy")
    
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
        _,\
        _,\
        _,\
        _, = data

    # average_rewards_buffer = average_per_steps(average_rewards_buffer, STEPS)
    # min_rewards_buffer = average_per_steps(min_rewards_buffer, STEPS)
    # max_rewards_buffer = average_per_steps(max_rewards_buffer, STEPS)

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



def average_per_steps(data, steps):
    assert steps <= len(data), 'Steps is larger that the size of the array'

    new_data = []
    for i in range(steps, len(data)+1):
        # print(data[i-steps: i])
        new_data.append(
            np.average(data[i-steps: i])
        )

    return new_data

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

if __name__ == "__main__":
    
    # plot_attentions()
    # tuner()

    data = [1,2,3,4,5,6]
    steps = 6

    average_per_steps(data, steps)