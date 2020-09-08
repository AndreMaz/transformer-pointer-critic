# For plotting
from environment.custom.knapsack.heuristic import solver
import matplotlib.pyplot as plt
import os
import numpy as np
# Import Google OR Tools Solver
# from agents.optimum_solver import solver


def plotter(data, env, agent, agent_config, opt_solver, print_details=False):
    
    # Destructure the tuple
    average_rewards_buffer, min_rewards_buffer, max_rewards_buffer = data

    # Compute optimum solution
    optimum_value = 0
    if opt_solver is not None:
        input_solver = env.convert_to_ortools_input()
        optimum_value = opt_solver(input_solver, print_details)
    else:
        optimum_value = env.optimum_value
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