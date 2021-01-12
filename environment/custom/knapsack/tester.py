from environment.custom.knapsack.env_v2 import KnapsackV2
from agents.agent import Agent
from environment.custom.knapsack.plotter import plot_attentions

# from agents.optimum_solver import solver
import numpy as np
import time


OPTIMAL = 'Optimal'
HEURISTIC = 'Heuristic'

def test(env: KnapsackV2, agent: Agent, opts: dict, opt_solver, heuristic_solver, look_for_opt: bool = False, show_info: bool = False):
    # for _ in range(32):
    # Set the agent to testing mode
    agent.training = False
    agent.stochastic_action_selection = False
    
    # Increase the number for resources during testing
    env.resource_sample_size = 20
    agent.num_resources = 20

    # Increase the number of bins during testing
    env.bin_sample_size = 5 + 1 # Because of the EOS
    
    # env.batch_size  = 1
    # agent.batch_size = 1

    training_step = 0
    isDone = False

    episode_rewards = np.zeros((agent.batch_size, agent.num_resources), dtype="float32")
    current_state, bin_net_mask, resource_net_mask, mha_used_mask = env.reset()
    dec_input = agent.generate_decoder_input(current_state)
    
    print(f'Testing with {agent.num_resources} resources and {env.bin_sample_size} bins')

    # Compute optimal values
    optimal_values = []
    if look_for_opt:
        print('Looking for Optimal Solutions...')
        start = time.time()
        for index in range(env.batch_size):
            print(f'Solving {index} of {env.batch_size}', end='\r')
            data = env.convert_to_ortools_input(index)
            optimal_values.append(opt_solver(data, False))
        print(f'Done! Optimal Solutions found in {time.time() - start:.2f} seconds')

    heuristic_values = []
    print('Looking for Heuristic Solutions...')
    start = time.time()
    for index in range(env.batch_size):
        print(f'Solving {index} of {env.batch_size}', end='\r')
        prob = env.batch[index]
        heuristic_values.append(heuristic_solver(prob, env.bin_sample_size))
    print(f'Done! Heuristic Solutions found in {time.time() - start:.2f} seconds')

    print('Solving with nets...')
    start = time.time()

    # attention_size = env.resource_sample_size + env.bin_sample_size
    # resource_attentions = np.zeros((env.resource_sample_size, attention_size), dtype="float32")
    # bin_attentions = np.zeros((env.resource_sample_size, attention_size), dtype="float32")
    
    attentions = []

    while not isDone:
        print(f'Placing step {training_step} of {agent.num_resources}', end='\r')
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

    if env.validate_history() == True:
        print('All solutions are valid!')
    else:
        print('Ups! Network generated invalid solutions')

    print(f'Done! Net solutions found in {time.time() - start:.2f} seconds')

    # print(episode_rewards)
    episode_rewards = np.sum(episode_rewards, axis=-1)
    
    if look_for_opt == False:
        optimal_values = len(episode_rewards) * [0]

    stats = zip(optimal_values, episode_rewards, heuristic_values)
    
    for opt_val, net_val, heu_val in stats:
        if look_for_opt == True:
            # Net to Opt distance
            d_from_opt_net = 100 - (net_val * 100 / opt_val)
            # Heuristic to Opt distance
            d_from_opt_heu = 100 - (heu_val * 100 / opt_val)

            print(f'Opt {opt_val} \t| Net {net_val} \t| % from Opt {d_from_opt_net:.2f} \t || Heuristic {heu_val} \t| % from Opt {d_from_opt_heu:.2f}')
        else:
            # Net to Heuristic Distance
            d_from_opt = 100 - (net_val * 100 / heu_val)

            print(f'Net {net_val} \t| Heuristic {heu_val} \t| % from Heuristic {d_from_opt:.2f}')

    # Plot the attentions to visualize the policy
    # plot_attentions(attentions, env.resource_sample_size, env.bin_sample_size)