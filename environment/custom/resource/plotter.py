# For plotting
# from environment.custom.knapsack.heuristic import solver
import matplotlib.pyplot as plt
import os
import numpy as np
# Import Google OR Tools Solver
# from agents.optimum_solver import solver


def plotter(data, env, agent, agent_config, opt_solver, print_details=False):
    
    # Destructure the tuple
    average_rewards_buffer, min_rewards_buffer, max_rewards_buffer, value_loss_buffer = data

    # Compute optimum solution
    optimum_value = 0
    if opt_solver is not None:
        input_solver = env.convert_to_ortools_input()
        optimum_value = opt_solver(input_solver, print_details)
    else:
        optimum_value = 0
    # Fill the array with the opt values
    # This will create a flat line
    opt_values = [optimum_value for i in range(len(average_rewards_buffer))]
    
    if env.name == 'CVRP':
        average_rewards_buffer = -1 * np.array(average_rewards_buffer)
        min_rewards_buffer = -1 * np.array(min_rewards_buffer)
        max_rewards_buffer = -1 * np.array(max_rewards_buffer)

    agent_name = agent.name
    env_name = env.name

    x_values = [i for i in range(len(average_rewards_buffer))]
    
    plt.plot(x_values, average_rewards_buffer, label="Average (in batch) Double Pointer Critic")
    plt.plot(x_values, min_rewards_buffer, label="Minimum (in Batch) Double Pointer Critic")
    plt.plot(x_values, max_rewards_buffer, label="Maximum (in batch) Double Pointer Critic")
    plt.plot(x_values, opt_values, label="Optimal")

    plt.ylabel('Collected Reward')
    plt.xlabel('Episode')

    gamma = agent_config['gamma']
    mha_mask = agent_config['use_mha_mask']
    entropy = agent_config['entropy_coefficient']
    td = agent_config['actor']['encoder_embedding_time_distributed']

    actor_lr = agent_config['actor']['learning_rate']
    critic_lr = agent_config['critic']['learning_rate']
    
    file_name = f"g:{gamma}|e:{entropy}|td:{td}|ac_lr:{actor_lr}|cr_lr:{critic_lr}|mha_mask:{mha_mask}"

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

def plot_attentions(attentions,
                    num_resources,
                    num_bins,
                    resource_normalization_factor,
                    task_normalization_factor):
    
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
        if int(round(resource_input[0,0,0])) == -1:
            CPU = -1
            RAM = -1
            MEM = -1
            task = -1
        else:
            CPU = int(round(resource_input[0,0,0]  * resource_normalization_factor))
            RAM = int(round(resource_input[0,0,1]  * resource_normalization_factor))
            MEM = int(round(resource_input[0,0,2]  * resource_normalization_factor))
            task = int(round(resource_input[0,0,3] * task_normalization_factor))

        req_type = int(round(resource_input[0,0,4]))

        resource_xlabel = f'C:{CPU} R:{RAM} M:{MEM} T:{task} P:{req_type}'
        plt.xticks([0], [resource_xlabel], fontsize=8)

        resource_states = attentions[index]['current_state'][0, num_bins:]
        resource_ylabel = []
        for itm in resource_states:
            if int(round(itm[0])) == -1:
                CPU = -1
                RAM = -1
                MEM = -1
                task = -1
            else:
                CPU = int(round(itm[0] * resource_normalization_factor)  )
                RAM = int(round(itm[1] * resource_normalization_factor)  )
                MEM = int(round(itm[2] * resource_normalization_factor)  )
                task = int(round(itm[3] * task_normalization_factor) )

            req_type = int(round(itm[4]))

            resource_ylabel.append(
                f'C:{CPU} R:{RAM} M:{MEM} T:{task} P:{req_type}'
            )

        plt.yticks(range(len(resource_ylabel)), resource_ylabel, rotation=0, fontsize=8)

        # Select the plot by index for the Backpacks
        plt.sca(axs[index, 1])
        # Add the ticks and the labels
        resource_input = attentions[index]["bin_net_input"]
        CPU = int(round(resource_input[0,0,0] * resource_normalization_factor)  )
        RAM = int(round(resource_input[0,0,1] * resource_normalization_factor)  )
        MEM = int(round(resource_input[0,0,2] * resource_normalization_factor)  )

        task = int(round(resource_input[0,0,3] * task_normalization_factor) )
        # upper_task = int(round(resource_input[0,0,4] * task_normalization_factor) )
        req_type = int(round(resource_input[0,0,4]))

        bin_xlabel = f'C:{CPU} R:{RAM} M:{MEM} T:{task} P:{req_type}'
        plt.xticks([0], [bin_xlabel], fontsize=8)

        bin_states = attentions[index]['current_state'][0, :num_bins]
        bin_ylabel = []
        for bp in bin_states:
            CPU = int(round(bp[0] * resource_normalization_factor)  )
            RAM = int(round(bp[1] * resource_normalization_factor)  )
            MEM = int(round(bp[2] * resource_normalization_factor)  )

            lower_task = int(round(bp[3] * task_normalization_factor) )
            upper_task = int(round(bp[4] * task_normalization_factor) )


            bin_ylabel.append(
                f'C:{CPU} R:{RAM} M:{MEM} L:{lower_task} U:{upper_task}'
            )
        plt.yticks(range(len(bin_ylabel)), bin_ylabel, rotation=0, fontsize=8)
    
    # plt.subplots_adjust(wspace=0.3, hspace = 0.3)
    plt.tight_layout()
    plt.show(block=True)

if __name__ == "__main__":
    
    plot_attentions()
    # tuner()