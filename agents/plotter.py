import matplotlib.pyplot as plt
import os
import numpy as np

def plotter(data, env, agent, agent_config: dict, trainer_config: dict, log_dir: str):
    
    export_stats: bool = trainer_config['export_stats']['export_stats']
    folder: bool = trainer_config['export_stats']['folder']

    if not export_stats: 
        return

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