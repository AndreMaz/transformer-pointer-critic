# For plotting
from environment.custom.knapsack.heuristic import solver
import matplotlib.pyplot as plt
import os

# Import Google OR Tools Solver
# from agents.optimum_solver import solver


def plotter(data, env, agent, agent_config, opt_solver, print_details=False):
    
    # Compute optimum solution
    input_solver = env.convert_to_ortools_input()
    optimum_value = opt_solver(input_solver, print_details)
    opt_values = [optimum_value for i in range(len(data))]

    agent_name = agent.name
    env_name = env.name

    x_values = [i for i in range(len(data))]
    
    plt.plot(x_values, data, label="Double Pointer Critic")
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