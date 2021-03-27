import numpy as np
import os
import time
from datetime import datetime


from environment.custom.knapsack_v2.env import KnapsackEnvironmentV2
from environment.custom.knapsack_v2.attention_plotter import attention_plotter
from agents.agent import Agent
from environment.custom.knapsack_v2.heuristic.factory import heuristic_factory
from environment.custom.knapsack_v2.misc.csv_writer import export_to_csv, log_testing_stats
from environment.custom.knapsack_v2.misc.utils import compute_max_steps, gather_stats_from_solutions, generate_file_name


def test(
    env: KnapsackEnvironmentV2,
    agent: Agent,
    opts: dict,
    log_dir: str
    ):

    
    num_tests: int = opts['testbed']['num_tests']
    # Number of bins
    bin_configs: dict = opts['testbed']['bin_sample_configs']
    bin_size_min = bin_configs['min']
    bin_size_max = bin_configs['max']
    bin_size_step = bin_configs['step']
    
    # Bins capacities
    bin_available_capacities: dict = opts['testbed']['bin_available_capacities']
    bin_min_resource = bin_available_capacities['min']
    bin_max_resource = bin_available_capacities['max']
    bin_step_resource = bin_available_capacities['step']

    # Number of items
    item_configs: dict = opts['testbed']['item_sample_configs']
    item_size_min = item_configs['min']
    item_size_max = item_configs['max']
    item_size_step = item_configs['step']
    
    batch_size: int = opts['batch_size']

    show_per_test_stats: bool = opts['show_per_test_stats']

    export_stats: bool = opts['export_stats']['global_stats']['export_stats']
    test_folder: str = opts['export_stats']['global_stats']['folder']
    
    filename: str = opts['export_stats']['global_stats']['filename']
    if filename == None:
        filename = generate_file_name(agent.agent_config)

    global_stats = []
    global_reward_results = np.array([
        0, # Won
        0, # Draw
        0, # Lost
    ])

    for item_sample_size in range(item_size_min, item_size_max+1, item_size_step):
        for bin_sample_size in range(bin_size_min, bin_size_max+1, bin_size_step):
            for bin_min_value in range(bin_min_resource, bin_max_resource, bin_step_resource):
                # print(f'{node_min_value}||{node_min_value + node_step_resource}')
                for index in range(num_tests):

                    instance_stats, reward_result = test_single_instance(
                        index,
                        env,
                        agent,
                        opts,
                        batch_size,
                        bin_sample_size, # Number of nodes
                        bin_min_value, # Min resources available in each node
                        bin_min_value + bin_step_resource, # Max resources available in each node
                        item_sample_size, # Number of resources
                        log_dir
                    )

                    global_reward_results += reward_result

                    global_stats.append({
                        "test_instance": index,
                        "bin_sample_size": bin_sample_size,
                        "bin_min_value": bin_min_value,
                        "bin_max_value": bin_min_value + bin_step_resource,
                        "item_sample_size": item_sample_size,
                        "instance": instance_stats
                    })

    if export_stats:
        f = os.path.join(log_dir, test_folder)
        if not os.path.isdir(f):
            os.makedirs(f)
        log_testing_stats(global_stats, f, filename)

    return global_reward_results

def test_single_instance(
    instance_id,
    env: KnapsackEnvironmentV2,
    agent: Agent,
    opts: dict,
    batch_size: int,
    bin_sample_size: int,
    bin_min_val: int,
    bin_max_val: int,
    item_sample_size: int,
    log_dir: str,
    ):
    
    plot_attentions: bool = opts['plot_attentions']

    # batch_size: int = opts['batch_size']
    # req_sample_size: int = opts['profiles_sample_size']
    # node_sample_size: int = opts['node_sample_size']

    export_stats: bool = opts['export_stats']['per_problem_stats']['export_stats']
    folder: str = opts['export_stats']['per_problem_stats']['folder']

    show_inference_progress: bool = opts['show_inference_progress']
    show_solutions: bool = opts['show_solutions']
    show_detailed_solutions: bool = opts['show_detailed_solutions']

    # Set the agent and env to testing mode
    env.set_testing_mode(
        batch_size,
        bin_sample_size,
        item_sample_size,
        bin_min_val,
        bin_max_val
    )
    agent.set_testing_mode(
        batch_size,
        env.bin_sample_size,
        env.item_sample_size
    )
    
    training_step = 0
    isDone = False
    episode_rewards = np.zeros(
        (agent.batch_size, agent.num_resources), dtype="float32")
    current_state, dec_input, bin_net_mask, mha_used_mask = env.reset()

    if show_inference_progress:
        print(f'Testing with {agent.num_resources} resources and {env.bin_sample_size} bins', end='\r')

    # Init the heuristic solvers 
    heuristic_solvers = heuristic_factory(env.bin_sample_size, opts['heuristic'])
    heuristic_input_state = current_state.copy()

    start = time.time()
    attentions = []

    while not isDone:
        if show_inference_progress:
            print(f'Placing step {training_step} of {agent.num_resources}', end='\r')

        # Select an action
        bin_id,\
        bin_net_mask,\
        bins_probs = agent.act(
            current_state,
            dec_input,
            bin_net_mask,
            mha_used_mask,
            env.build_feasible_mask
        )


        next_state, next_dec_input, reward, isDone, info = env.step(
            bin_id,
            bin_net_mask
        )
                
        # Store episode rewards
        episode_rewards[:, training_step] = reward[:, 0]

        attentions.append({
            "current_state": current_state.copy(),
            'decoder_input': np.array(dec_input),
            "attention_probs": bins_probs.numpy()
        })

        # Update for next iteration
        current_state = next_state
        dec_input = next_dec_input.copy()
        bin_net_mask = info['bin_net_mask']
        mha_used_mask = info['mha_used_mask']
        
        training_step += 1

    if show_inference_progress:
        # if env.validate_history() == True:
        #    print('All solutions are valid!')
        #else:
        #    print('Ups! Network generated invalid solutions')

        print(f'Net solution found in {time.time() - start:.2f} seconds', end='\r')
    
    # Solve with Heuristic
    for solver in heuristic_solvers:
        solver.solve(heuristic_input_state)
    
    if export_stats:
        # Find the the node with maximum number of inserted resources
        max_steps = compute_max_steps(env.history[0], heuristic_solvers)
        t = datetime.now().replace(microsecond=0).isoformat()
        # Export results to CSV
        f = os.path.join(log_dir, folder)
        if not os.path.isdir(f):
            os.makedirs(f)

        export_to_csv(env.history, max_steps, agent.name, f'{f}/{t}_{instance_id}')
        for solver in heuristic_solvers:
            export_to_csv([solver.solution], max_steps, solver.name, f'{f}/{t}_{instance_id}')


    if show_solutions:
        env.print_history(show_detailed_solutions)
        print('________________________________________________________________________________')    
        for solver in heuristic_solvers:
            solver.print_node_stats(show_detailed_solutions)

    if plot_attentions:
        # Plot the attentions to visualize the policy
        attention_plotter(
            attentions,
            env.item_sample_size,
            env.bin_sample_size,
        )
    
    stats, reward_result = gather_stats_from_solutions(env, heuristic_solvers)

    return stats, reward_result