from agents.agent import Agent
from agents.tester import test as tester
from agents.plotter import plotter
from agents.trainer import trainer

from environment.env_factory import env_factory
from configs.configs import get_configs
# For params tunning
import numpy as np
import math

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)

def runner(env_type="custom", env_name='KnapsackV2', agent_name="tpc"):
    # Read the configs
    agent_config, trainer_config, env_config = get_configs(env_name, agent_name)

    # Create the environment
    env = env_factory(env_type, env_name, env_config)

    # Add info about the environment
    agent_config = env.add_stats_to_agent_config(agent_config)
    
    # Create the agent
    agent = Agent('transformer', agent_config)

    # Train
    print('Training...')
    training_history = trainer(env, agent, trainer_config)
    
    # Plot the learning curve
    print('\nPlotting Results...')
    plotter(training_history, env, agent, agent_config, False)

    # Test the agent
    print("\nTesting...")
    tester(env, agent)
    print('End... Goodbye!')

def tuner(env_type="custom", env_name='KnapsackV2', agent_name="tpc"):
    # Read the configs
    agent_config, trainer_config, env_config = get_configs(env_name, agent_name)
    
    # Create the environment
    env = env_factory(env_type, env_name, env_config)
    
    # Add info about the environmanet
    agent_config: dict = env.add_stats_to_agent_config(agent_config)

    entropy_coefficient = [ 0.00001, 0.0001, 0.001 ]
    actor_learning_rate = [ 0.00001, 0.0001, 0.0005 ]
    critic_learning_rate = [ 0.00001, 0.0001, 0.0005 ]
    mha_mask = [ True, False ]
    time_distributed = [
        True,
        # False
    ]

    for entropy in entropy_coefficient:
        for actor_lr in actor_learning_rate:
            for critic_lr in critic_learning_rate:
                for td in time_distributed:
                    for use_mask in mha_mask:
                    
                        # Swap between using and not using the MHA mask
                        env.compute_mha_mask = use_mask

                        config = agent_config.copy()

                        config['entropy_coefficient'] = entropy

                        config['actor']['learning_rate'] = actor_lr
                        config['actor']['encoder_embedding_time_distributed'] = td

                        config['critic']['learning_rate'] = critic_lr
                        config['critic']['encoder_embedding_time_distributed'] = td

                        config['mha_mask'] = use_mask

                        agent = Agent('transformer', config)

                        training_history = trainer(
                            env, agent, trainer_config)

                        plotter(training_history, env,
                                agent, config, False)



if __name__ == "__main__":
    tuner()
    # runner()
