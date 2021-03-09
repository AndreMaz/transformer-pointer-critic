import numpy as np
from environment.custom.resource_v3.env import ResourceEnvironmentV3
from agents.agent import Agent
from environment.custom.resource_v3.plotter import plot_attentions
from environment.custom.resource_v3.utils import export_to_csv, compute_max_steps, gather_stats_from_solutions, log_testing_stats

# from agents.optimum_solver import solver
from environment.custom.resource_v3.heuristic.factory import heuristic_factory
from environment.custom.resource_v3.utils import generate_file_name
import os
import time
from datetime import datetime

OPTIMAL = 'Optimal'
HEURISTIC = 'Heuristic'


def test(
    env: ResourceEnvironmentV3,
    agent: Agent,
    opts: dict,
    log_dir: str
    ):

    num_tests: int = opts['testbed']['num_tests']
    node_configs: dict = opts['testbed']['node_configs']
    node_size_min = node_configs['min']
    node_size_max = node_configs['max']
    node_size_step = node_configs['step']

    node_available_resource: dict = opts['testbed']['node_available_resources']
    node_min_resource = node_available_resource['min']
    node_max_resource = node_available_resource['max']
    node_step_resource = node_available_resource['step']

    resource_configs: dict = opts['testbed']['request_configs']
    resource_size_min = resource_configs['min']
    resource_size_max = resource_configs['max']
    resource_size_step = resource_configs['step']
    
    batch_size: int = opts['batch_size']

    show_per_test_stats: bool = opts['show_per_test_stats']

    export_stats: bool = opts['export_stats']['global_stats']['export_stats']
    test_folder: str = opts['export_stats']['global_stats']['folder']
    
    filename: str = opts['export_stats']['global_stats']['filename']
    if filename == None:
        filename = generate_file_name(agent.agent_config)

    global_stats = []
    dominant_results = np.array([
        0, # Won
        0, # Draw
        0, # Lost
    ])

    rejected_results = np.array([
        0, # Won
        0, # Draw
        0, # Lost
    ])

    for resource_sample_size in range(resource_size_min, resource_size_max+1, resource_size_step):
        for node_sample_size in range(node_size_min, node_size_max+1, node_size_step):
            for node_min_value in range(node_min_resource, node_max_resource, node_step_resource):
                # print(f'{node_min_value}||{node_min_value + node_step_resource}')
                for index in range(num_tests):

                    instance_stats,\
                    dominant_instance_result,\
                    rejected_instance_result = test_single_instance(
                        index,
                        env,
                        agent,
                        opts,
                        batch_size,
                        node_sample_size, # Number of nodes
                        node_min_value, # Min resources available in each node
                        node_min_value + node_step_resource, # Max resources available in each node
                        resource_sample_size, # Number of resources
                        log_dir
                    )

                    dominant_results += dominant_instance_result
                    rejected_results += rejected_instance_result

                    global_stats.append({
                        "test_instance": index,
                        "node_sample_size": node_sample_size,
                        "node_min_value": node_min_value,
                        "node_max_value": node_min_value + node_step_resource,
                        "resource_sample_size": resource_sample_size,
                        "instance": instance_stats
                    })

    if export_stats:
        f = os.path.join(log_dir, test_folder)
        if not os.path.isdir(f):
            os.makedirs(f)
        log_testing_stats(global_stats, f, filename)

    return dominant_results, rejected_results

def test_single_instance(
    instance_id,
    env: ResourceEnvironmentV3,
    agent: Agent,
    opts: dict,
    batch_size: int,
    node_sample_size: int,
    node_min_val: int,
    node_max_val: int,
    req_sample_size: int,
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
        node_sample_size,
        req_sample_size,
        node_min_val,
        node_max_val
    )
    agent.set_testing_mode(
        batch_size,
        node_sample_size,
        req_sample_size
    )
    
    training_step = 0
    isDone = False
    episode_rewards = np.zeros(
        (agent.batch_size, agent.num_resources), dtype="float32")
    current_state, dec_input, bin_net_mask, mha_used_mask = env.reset()

    if show_inference_progress:
        print(f'Testing with {agent.num_resources} resources and {env.node_sample_size} bins', end='\r')

    # Init the heuristic solvers 
    heuristic_solvers = heuristic_factory(env.node_sample_size, opts['heuristic'])
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
            'resource_net_input': np.array(dec_input),
            "bin_attention": bins_probs.numpy(),
            "current_state": current_state.copy()
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
        plot_attentions(
            attentions,
            env.profiles_sample_size,
            env.node_sample_size,
        )
    
    stats,\
        dominant_result,\
        rejected_result = gather_stats_from_solutions(env, heuristic_solvers)

    return stats, dominant_result, rejected_result