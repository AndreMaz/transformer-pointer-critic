import numpy as np
from environment.custom.resource_v2.env import ResourceEnvironmentV2
from agents.agent import Agent
from environment.custom.resource_v2.plotter import plot_attentions
from environment.custom.resource_v2.utils import export_to_csv, compute_max_steps, compute_delta, num_overloaded_nodes


# from agents.optimum_solver import solver
import numpy as np
import time
from datetime import datetime

OPTIMAL = 'Optimal'
HEURISTIC = 'Heuristic'


def test(
    env: ResourceEnvironmentV2,
    agent: Agent,
    opts: dict,
    opt_solver,
    heuristic_solver,
    look_for_opt: bool = False,
    show_info: bool = False
    ):

    num_tests = opts['num_tests']
    show_info = opts['show_info']
    
    # print('Problem;Net_Total;Net_Free;Net_Premium;Net_Free_Batch;Net_Premium_Batch;Heu_Total;Net_Free;Heu_Premium;Heu_Free_Batch;Heu_Premium_Batch')
    
    totals = 0
    for i in range(num_tests):
        env, solver = test_single_instance(
            i,
            env,
            agent,
            opts,
            opt_solver,
            heuristic_solver,
            show_info=show_info
        )

        net_delta = compute_delta(env.history[0])
        heu_delta = compute_delta(solver.node_list)
        pos = np.argmax([net_delta, heu_delta])
        if pos == 0:
            totals += 1

        # print(f'{net_delta[0]:.5f};{heu_delta[0]:.5f};{pos}')
    
    # print(f"Net won in {totals/num_tests}%")

    return totals/num_tests

def test_single_instance(
    instance_id,
    env: ResourceEnvironmentV2,
    agent: Agent,
    opts: dict,
    opt_solver,
    heuristic_solver,
    look_for_opt: bool = False,
    show_info: bool = False
    ):
    
    batch_size = opts['batch_size']
    req_sample_size = opts['profiles_sample_size']
    node_sample_size = opts['node_sample_size']
    num_episodes = opts['num_episodes']
    csv_write_path = opts['export_stats']['location']

    # Set the agent and env to testing mode
    env.set_testing_mode(batch_size, node_sample_size, req_sample_size)
    agent.set_testing_mode(batch_size, req_sample_size)
    
    episode_count = 0
    training_step = 0
    isDone = False

    episode_rewards = np.zeros((agent.batch_size, agent.num_resources * num_episodes), dtype="float32")
    current_state, bin_net_mask, resource_net_mask, mha_used_mask = env.reset()
    dec_input = agent.generate_decoder_input(current_state)
    
    env.build_history(current_state)

    if show_info:
        print(f'Testing for {num_episodes} episodes with {agent.num_resources} resources and {env.node_sample_size} bins')

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
            if show_info:
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

    if show_info:
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
    export_to_csv(env.history, max_steps, 'Neural', f'{csv_write_path}/{t}_{instance_id}_net.csv')
    export_to_csv([solver.node_list], max_steps, 'Heuristic', f'{csv_write_path}/{t}_{instance_id}_heuristic.csv')


    if show_info:
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