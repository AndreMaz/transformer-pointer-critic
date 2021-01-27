import numpy as np
from environment.custom.resource_v3.env import ResourceEnvironmentV3
from agents.agent import Agent
from environment.custom.resource_v3.plotter import plot_attentions
from environment.custom.resource_v3.utils import export_to_csv, compute_max_steps, compute_delta, round_half_up

# from agents.optimum_solver import solver
from environment.custom.resource_v3.heuristic.factory import heuristic_factory
import numpy as np
import time
from datetime import datetime

OPTIMAL = 'Optimal'
HEURISTIC = 'Heuristic'


def test(
    env: ResourceEnvironmentV3,
    agent: Agent,
    opts: dict,
    ):

    num_tests = opts['num_tests']
    show_per_test_stats = opts['show_per_test_stats']

    
    won = 0
    draw = 0
    loss = 0
    for test_index in range(num_tests):
        env, solver = test_single_instance(
            test_index,
            env,
            agent,
            opts,
        )

        net_delta, net_rejected = compute_delta(env.history[0])
        heu_delta, heu_rejected = compute_delta(solver.solution)
        
        # Round to 2 decimals
        net_delta = round_half_up(net_delta, 2)
        heu_delta = round_half_up(heu_delta, 2)

        if net_delta > heu_delta:
            won += 1
            res = 'Won'
        elif net_delta == heu_delta:
            draw += 1
            res = 'Draw'
        else:
            loss += 1
            res = 'Loss'

        if show_per_test_stats:
            print(f'{net_delta[0]:.2f};{net_rejected}||{heu_delta[0]:.2f};{heu_rejected}||{res}')
    
    if show_per_test_stats:
        print(f"Won {(won/num_tests)*100}% || Draw {(draw/num_tests)*100}% || Loss {(loss/num_tests)*100}%")

    return won/num_tests, draw/ num_tests, loss / num_tests

def test_single_instance(
    instance_id,
    env: ResourceEnvironmentV3,
    agent: Agent,
    opts: dict
    ):
    
    plot_attentions = opts['plot_attentions']
    batch_size = opts['batch_size']
    req_sample_size = opts['profiles_sample_size']
    node_sample_size = opts['node_sample_size']
    num_episodes = opts['num_episodes']

    csv_write_path = opts['export_stats']['location']
    export_stats = opts['export_stats']['export_stats']

    show_inference_progress = opts['show_inference_progress']
    show_solutions = opts['show_solutions']
    show_detailed_solutions = opts['show_detailed_solutions']

    # Set the agent and env to testing mode
    env.set_testing_mode(batch_size, node_sample_size, req_sample_size)
    agent.set_testing_mode(batch_size, req_sample_size)
    
    episode_count = 0
    training_step = 0
    isDone = False

    episode_rewards = np.zeros((agent.batch_size, agent.num_resources * num_episodes), dtype="float32")
    current_state, bin_net_mask, resource_net_mask, mha_used_mask = env.reset()
    dec_input = agent.generate_decoder_input(current_state)
    
    if show_inference_progress:
        print(f'Testing for {num_episodes} episodes with {agent.num_resources} resources and {env.node_sample_size} bins')

    # print('Solving with nets...')
    start = time.time()

    attentions = []

    # Init the heuristic solvers 
    # solver = heuristic_solvers(env.node_sample_size, opts['heuristic']['greedy'])
    heuristic_solvers = heuristic_factory(node_sample_size, opts['heuristic'])
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
            if show_inference_progress:
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

    if show_inference_progress:
        # if env.validate_history() == True:
        #    print('All solutions are valid!')
        #else:
        #    print('Ups! Network generated invalid solutions')

        print(f'Done! Net solutions found in {time.time() - start:.2f} seconds')
    
    # Solve with Heuristic
    for state in state_list:
        for solver in heuristic_solvers:
            solver.solve(state)
    
    if export_stats:
        # Find the the node with maximum number of inserted resources
        max_steps = compute_max_steps(env.history[0], heuristic_solvers)
        # Export results to CSV
        t = datetime.now().replace(microsecond=0).isoformat()
        export_to_csv(env.history, max_steps, agent.name, f'{csv_write_path}/{t}_{instance_id}')
        for solver in heuristic_solvers:
            export_to_csv([solver.solution], max_steps, solver.name, f'{csv_write_path}/{t}_{instance_id}')


    if show_solutions:
        env.print_history(show_detailed_solutions)
        print('________________________________________________________________________________')    
        for solver in heuristic_solvers:
            solver.print_node_stats(show_detailed_solutions)

    if plot_attentions:
        # Plot the attentions to visualize the policy
        plot_attentions(
            attentions,
            env.profiles_sample_size,
            env.node_sample_size,
        )

    return env, solver