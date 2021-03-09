from tensorflow.python.ops.gen_math_ops import log
from agents.agent import Agent
from agents.trainer import trainer

from environment.env_factory import env_factory
from configs.configs import get_configs
# For params tunning
import json
import os
from datetime import datetime

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

LOG_DIR = "./results/"

def runner(env_type="custom", env_name='ResourceV3', agent_name="tpc"):

    # Store the time of the script
    start_date = datetime.now().replace(microsecond=0).isoformat()
    
    # Read the configs
    agent_config, trainer_config, env_config, tester_config, _, all_configs = get_configs(env_name, agent_name)

    # Create the environment
    env, tester, plotter = env_factory(env_type, env_name, env_config)

    # Add info about the environment
    agent_config = env.add_stats_to_agent_config(agent_config)
    
    # Create the agent
    agent = Agent('transformer', agent_config)

    # Create a dir for logging training and testing results
    log_dir = os.path.join(LOG_DIR, env.name, start_date)
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    # Store all configs used to train the model
    json.dump(all_configs, open(os.path.join(log_dir, "config.json"), "w"), indent = 6)
    
    # Store the environment data for reproducibility
    env.store_dataset(os.path.join(log_dir, "env.txt"))

    # Train
    print('Training...')
    # tf.profiler.experimental.start('logdir')
    show_progress = True
    training_history = trainer(env, agent, trainer_config, show_progress, log_dir)
    # tf.profiler.experimental.stop()

    print('\nTraining Done...')

    # Plot training results (learning curve and rewards)
    print('\nPlotting Results...')
    plotter(training_history, env, agent, agent_config, trainer_config, log_dir)

    # Test the agent
    print("\nTesting...")
    tester(env, agent, tester_config, log_dir)
    print('\nEnd... Goodbye!')

def tuner(env_type="custom", env_name='ResourceV3', agent_name="tpc"):
    
    entropy_coefficient = [ 0.01, 0.02, 0.1 ]
    actor_layers = [
        1,
        # 2
    ]
    actor_learning_rate = [
        0.0001, 0.0002
    ]
    critic_learning_rate = [ 
        0.0001, 0.0002
    ]

    for actor_layer in actor_layers:
        for entropy in entropy_coefficient:
            for actor_lr in actor_learning_rate:
                for critic_lr in critic_learning_rate:
                        # Read the configs
                        agent_config, trainer_config, env_config, _, tuner_config = get_configs(env_name, agent_name)

                        # Create the environment
                        env, tester, plotter = env_factory(env_type, env_name, env_config)

                        # Add info about the environmanet
                        agent_config: dict = env.add_stats_to_agent_config(agent_config)


                        config = agent_config.copy()

                        config['entropy_coefficient'] = entropy

                        config['actor']['num_layers'] = actor_layer
                        config['actor']['learning_rate'] = actor_lr

                        config['critic']['learning_rate'] = critic_lr

                        agent = Agent('transformer', config)

                        show_info = False
                        training_history = trainer(
                            env, agent, trainer_config, show_info)

                        write_data_to_file = True
                        plotter(training_history, env, agent, config, write_data_to_file)

                        look_for_opt = False
                        dominant_results, rejected_results = tester(env, agent, tuner_config)

                        print(f"{dominant_results};{rejected_results};{actor_layer};{entropy};{actor_lr};{critic_lr}")

if __name__ == "__main__":
    runner()
    # tuner()
