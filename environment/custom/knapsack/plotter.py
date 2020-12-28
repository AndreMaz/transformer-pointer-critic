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

def plot_attentions(attentions, num_items, num_backpacks):
    
    fig, axs = plt.subplots(num_items, 2)

    for index, attention in enumerate(attentions):
        
        # Only show the attention over the items
        axs[index, 0].matshow(attention['item_attention'][:, num_backpacks:])
        # axs[index, 0].set_title('Item Attention')

        # Only show the attention over the backpacks
        axs[index, 1].matshow(attention['backpack_attention'][:, :num_backpacks])
        # axs[index, 1].set_title('Backpack Attention')

    for index in range(num_items):
        # Select the plot by index for the Items
        plt.sca(axs[index, 0])
        # Add the ticks and the labels
        item_input = attentions[index]["item_net_input"]
        item_ylabel = f'w:{int(item_input[0,0,0])} v:{int(item_input[0,0,1])}'
        plt.yticks([0], [item_ylabel])

        item_states = attentions[index]['current_state'][0, num_backpacks:]
        item_xlabel = []
        for itm in item_states:
            item_xlabel.append(
                f'w:{int(itm[0])} v:{int(itm[1])}'
            )
        plt.xticks(range(len(item_xlabel)), item_xlabel)

        # Select the plot by index for the Backpacks
        plt.sca(axs[index, 1])
        # Add the ticks and the labels
        item_input = attentions[index]["backpack_net_input"]
        backpack_ylabel = f'w:{int(item_input[0,0,0])} v:{int(item_input[0,0,1])}'
        plt.yticks([0], [backpack_ylabel])

        backpack_states = attentions[index]['current_state'][0, :num_backpacks]
        backpack_xlabel = []
        for bp in backpack_states:
            backpack_xlabel.append(
                f'c:{int(bp[0])} l:{int(bp[1])}'
            )
        plt.xticks(range(len(backpack_xlabel)), backpack_xlabel)
    
    # plt.subplots_adjust(wspace=0.3, hspace = 0.3)

    plt.show(block=True)

if __name__ == "__main__":
    
    plot_attentions()
    # tuner()