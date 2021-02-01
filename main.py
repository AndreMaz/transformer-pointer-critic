from agents.agent import Agent
from agents.trainer import trainer

from environment.env_factory import env_factory
from configs.configs import get_configs
# For params tunning
import numpy as np
import math

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# import tensorflow as tf
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#   # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
#   try:
#     tf.config.experimental.set_virtual_device_configuration(
#         gpus[0],
#         [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120)])
#     logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#     print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#   except RuntimeError as e:
#     # Virtual devices must be set before GPUs have been initialized
#     print(e)

def runner(env_type="custom", env_name='ResourceV3', agent_name="tpc"):
    # Read the configs
    agent_config, trainer_config, env_config, tester_config, _ = get_configs(env_name, agent_name)

    # Create the environment
    env, tester, plotter = env_factory(env_type, env_name, env_config)

    # Add info about the environment
    agent_config = env.add_stats_to_agent_config(agent_config)
    
    # Create the agent
    agent = Agent('transformer', agent_config)

    # Train
    print('Training...')
    # tf.profiler.experimental.start('logdir')
    show_progress = True
    training_history = trainer(env, agent, trainer_config, show_progress)
    # tf.profiler.experimental.stop()

    # Plot the learning curve
    print('\nPlotting Results...')
    write_data_to_file = True
    plotter(training_history, env, agent, agent_config, write_data_to_file)

    # Test the agent
    print("\nTesting...")
    tester(env, agent, tester_config)
    print('End... Goodbye!')

def tuner(env_type="custom", env_name='ResourceV3', agent_name="tpc"):
    
    gamma_rate = [0.99, 0.999]
    entropy_coefficient = [ 0.001, 0.01 ]
    dropout_rate = [ 0.001, 0.01, 0.1 ]
    actor_learning_rate = [
        0.00001,
        0.0005,
        #0.0005
    ]
    critic_learning_rate = [ 
        0.00001,
        0.0005,
        #0.0005
    ]

    for gamma in gamma_rate:
        for entropy in entropy_coefficient:
            for actor_lr in actor_learning_rate:
                for critic_lr in critic_learning_rate:
                        for dp_rate in dropout_rate:
                            # Read the configs
                            agent_config, trainer_config, env_config, _, tuner_config = get_configs(env_name, agent_name)

                            # Create the environment
                            env, tester, plotter = env_factory(env_type, env_name, env_config)

                            # Add info about the environmanet
                            agent_config: dict = env.add_stats_to_agent_config(agent_config)


                            config = agent_config.copy()

                            config['gamma'] = gamma
                            config['entropy_coefficient'] = entropy

                            config['actor']['learning_rate'] = actor_lr
                            config['actor']['dropout_rate'] = dp_rate

                            config['critic']['learning_rate'] = critic_lr
                            config['critic']['dropout_rate'] = dp_rate

                            agent = Agent('transformer', config)

                            show_info = False
                            training_history = trainer(
                                env, agent, trainer_config, show_info)

                            write_data_to_file = True
                            plotter(training_history, env, agent, agent_config, write_data_to_file)

                            look_for_opt = False
                            dominant_results, rejected_results = tester(env, agent, tuner_config)

                            print(f"{dominant_results};{rejected_results};{gamma};{entropy};{dp_rate};{actor_lr};{critic_lr}")

if __name__ == "__main__":
    # runner()
    tuner()
