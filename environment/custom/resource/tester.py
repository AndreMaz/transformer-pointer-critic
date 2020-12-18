from environment.custom.resource.env import ResourceEnvironment
from agents.agent import Agent
from environment.custom.resource.plotter import plot_attentions
from environment.custom.resource.utils import export_to_csv, compute_max_steps


# from agents.optimum_solver import solver
import numpy as np
import time
from datetime import datetime

OPTIMAL = 'Optimal'
HEURISTIC = 'Heuristic'

def test(env: ResourceEnvironment, agent: Agent, opts: dict, opt_solver, heuristic_solver, look_for_opt: bool = False):
    # for _ in range(32):
    
    # Set the agent to testing mode
    agent.training = False
    agent.stochastic_action_selection = False
    
    # Set the number for resources during testing
    env.resource_sample_size = opts['resource_sample_size']
    agent.num_resources = opts['resource_sample_size']

    # Set the number of bins during testing
    # + 1 Because of the EOS
    env.bin_sample_size = opts['bin_sample_size'] + 1

    num_episodes = opts['num_episodes']
    
    num_iterations_before_node_reset = opts['num_iterations_before_node_reset']
    env.num_iterations_before_node_reset = num_iterations_before_node_reset

    env.reset_num_iterations() # Reset the env
    env.batch_size  = 1
    agent.batch_size = 1
    
    episode_count = 0
    training_step = 0
    isDone = False

    episode_rewards = np.zeros((agent.batch_size, agent.num_resources * num_episodes), dtype="float32")
    current_state, bin_net_mask, resource_net_mask, mha_used_mask = env.reset()
    dec_input = agent.generate_decoder_input(current_state)
    
    # Allow nodes to gather the stats
    env.rebuild_history()
    env.set_testing_mode()

    print(f'Testing for {num_episodes} episodes with {agent.num_resources} resources and {env.bin_sample_size} bins')

    # print('Solving with nets...')
    start = time.time()

    attentions = []

    # Init the heuristic solver
    # This will parse nodes/bins
    solver = heuristic_solver(env, opts['heuristic']['greedy'])
    state_list = [current_state]

    while episode_count < num_episodes:
        # Reached the end of episode. Reset for the next episode
        if isDone:
            isDone = False
            current_state, bin_net_mask, resource_net_mask, mha_used_mask = env.reset()
            # SOS input for resource Ptr Net
            dec_input = agent.generate_decoder_input(current_state)
            training_step = 0

            # Store the states for the heuristic
            state_list.append(current_state)

        while not isDone:
            print(f'Episode {episode_count} Placing step {training_step} of {agent.num_resources}', end='\r')
            # Select an action
            bin_id,\
            resource_id,\
            decoded_resource,\
            bin_net_mask,\
            resources_probs,\
            bins_probs = agent.act(
                current_state,
                dec_input,
                bin_net_mask,
                resource_net_mask,
                mha_used_mask,
                env.build_feasible_mask
            )

            # Play one step
            next_state, reward, isDone, info = env.step(
                bin_id,
                resource_id,
                bin_net_mask
            )
                    
            # Store episode rewards
            episode_rewards[:, training_step] = reward[:, 0]

            attentions.append({
                'resource_net_input': np.array(dec_input),
                'bin_net_input': decoded_resource.numpy(),
                'resource_attention': resources_probs.numpy(),
                "bin_attention": bins_probs.numpy(),
                "current_state": current_state.copy()
            })

            # Update for next iteration
            dec_input = decoded_resource
            current_state = next_state
            bin_net_mask = info['bin_net_mask']
            resource_net_mask = info['resource_net_mask']
            mha_used_mask = info['mha_used_mask']
            
            training_step += 1

        episode_count += 1

    if env.validate_history() == True:
        print('All solutions are valid!')
    else:
        print('Ups! Network generated invalid solutions')

    print(f'Done! Net solutions found in {time.time() - start:.2f} seconds')
    
    # Solve with Heuristic
    for state in state_list:
         solver.solve(state)
    
    # Find the the node with maximum number of inserted resources
    max_steps = compute_max_steps(env.history[0], solver.node_list)
    # Export results to CSV
    t = datetime.now().replace(microsecond=0).isoformat()
    export_to_csv(env.history, max_steps, 'Neural', f'./results/resource/net_{t}.csv')
    export_to_csv([solver.node_list], max_steps, 'Heuristic', f'./results/resource/heuristic_{t}.csv')


    env.print_history(False)
    print('________________________________________________________________________________')    
    solver.print_node_stats(False)

    # print(episode_rewards)
    episode_rewards = np.sum(episode_rewards, axis=-1)
    
    if look_for_opt == False:
        optimal_values = len(episode_rewards) * [0]

    # Plot the attentions to visualize the policy
    # plot_attentions(
    #     attentions,
    #     env.resource_sample_size,
    #     env.bin_sample_size,
    #     env.resource_normalization_factor,
    #     env.task_normalization_factor
    # )

    return env, solver

def compute_opt_solutions(env: ResourceEnvironment, opt_solver):
    optimal_values = []
    print('Looking for Optimal Solutions...')
    start = time.time()
    for index in range(env.batch_size):
        print(f'Solving {index} of {env.batch_size}', end='\r')
        data = env.convert_to_ortools_input(index)
        optimal_values.append(opt_solver(data, False))
    print(f'Done! Optimal Solutions found in {time.time() - start:.2f} seconds')

    return optimal_values


def compute_heuristic_solutions(env: ResourceEnvironment, heuristic_solver):
    heuristic_values = []

    print('Looking for Heuristic Solutions...')
    start = time.time()
    for index in range(env.batch_size):
        print(f'Solving {index} of {env.batch_size}', end='\r')
        prob = env.batch[index]
        heuristic_values.append(heuristic_solver(prob, env.bin_sample_size))
    print(f'Done! Heuristic Solutions found in {time.time() - start:.2f} seconds')

    return heuristic_values