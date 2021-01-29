# For plotting
# from environment.custom.knapsack.heuristic import solver
import matplotlib.pyplot as plt
import os
import numpy as np
# Import Google OR Tools Solver
# from agents.optimum_solver import solver

STEPS = 100

def plotter(data, env, agent, agent_config, print_details=False):
    plotter_leaning(data, env, agent, agent_config, print_details)

    plotter_rewards(data, env, agent, agent_config, print_details)

def plotter_leaning(data, env, agent, agent_config, print_details=False):
# Destructure the tuple
    _,\
        _,\
        _,\
        value_loss_buffer, \
        resources_loss_buffer,\
        resources_entropy_buffer,\
        bins_loss_buffer,\
        bins_entropy_buffer = data
    
    value_loss_buffer = average_per_steps(value_loss_buffer, STEPS)
    resources_loss_buffer = average_per_steps(resources_loss_buffer, STEPS)
    resources_entropy_buffer = average_per_steps(resources_entropy_buffer, STEPS)
    bins_loss_buffer = average_per_steps(bins_loss_buffer, STEPS)
    bins_entropy_buffer = average_per_steps(bins_entropy_buffer, STEPS)

    x_values = [i for i in range(len(value_loss_buffer))]

    agent_name = agent.name
    env_name = env.name
    
    plt.plot(x_values, value_loss_buffer, label="Value Loss")
    plt.plot(x_values, resources_loss_buffer, label="Resource Net Loss")
    plt.plot(x_values, resources_entropy_buffer, label="Resource Net Entropy")
    plt.plot(x_values, bins_loss_buffer, label="Bin Net Loss")
    plt.plot(x_values, bins_entropy_buffer, label="Bin Net Entropy")

    plt.xlabel('Episode')

    file_name = generate_file_name(agent_config) + 'learning'

    plot_title = f"{agent_name.upper()}\n|" + file_name
    plt.title(plot_title)

    saveDir = f"./media/plots/{env_name}/{agent_name}"
    # Check if dir exists. If not, create it
    if not os.path.isdir(saveDir):
        os.makedirs(saveDir)

    # Show legend info
    plt.legend()

    # plt.show(block=blockPlot)
    plt.savefig(
        f"{saveDir}/{file_name.replace(' ', '')}.png",
        dpi = 200,
        bbox_inches = "tight"
    )

    plt.close()
    
def plotter_rewards(data, env, agent, agent_config, print_details=False):
    
    # Destructure the tuple
    average_rewards_buffer,\
        min_rewards_buffer,\
        max_rewards_buffer,\
        _, \
        _,\
        _,\
        _,\
        _ = data

    average_rewards_buffer = average_per_steps(average_rewards_buffer, STEPS)
    min_rewards_buffer = average_per_steps(min_rewards_buffer, STEPS)
    max_rewards_buffer = average_per_steps(max_rewards_buffer, STEPS)

    agent_name = agent.name
    env_name = env.name

    x_values = [i for i in range(len(average_rewards_buffer))]
    
    plt.plot(x_values, average_rewards_buffer, label="Average (in batch) Double Pointer Critic")
    plt.plot(x_values, min_rewards_buffer, label="Minimum (in Batch) Double Pointer Critic")
    plt.plot(x_values, max_rewards_buffer, label="Maximum (in batch) Double Pointer Critic")

    plt.ylabel('Collected Reward')
    plt.xlabel('Episode')

    file_name = generate_file_name(agent_config) + 'reward'

    plot_title = f"{agent_name.upper()}\n|" + file_name
    plt.title(plot_title)

    saveDir = f"./media/plots/{env_name}/{agent_name}"
    # Check if dir exists. If not, create it
    if not os.path.isdir(saveDir):
        os.makedirs(saveDir)

    # Show legend info
    plt.legend()

    # plt.show(block=blockPlot)
    plt.savefig(
        f"{saveDir}/{file_name.replace(' ', '')}.png",
        dpi = 200,
        bbox_inches = "tight"
    )

    plt.close()

def generate_file_name(agent_config):
    gamma = agent_config['gamma']
    entropy = agent_config['entropy_coefficient']
    dp_rate = agent_config['actor']['dropout_rate']
    
    actor_lr = agent_config['actor']['learning_rate']
    critic_lr = agent_config['critic']['learning_rate']

    file_name = f"g:{gamma}|e:{entropy}|dp:{dp_rate}|ac_lr:{actor_lr}|cr_lr:{critic_lr}"

    return file_name

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