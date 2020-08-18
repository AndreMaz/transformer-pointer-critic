from environment.custom.knapsack.env_v2 import KnapsackV2
from agents.agent import Agent

from agents.optimum_solver import solver
import numpy as np

def test(env: KnapsackV2, agent: Agent):
    # Compute optimal values
    optimal_values = []
    for index in range(env.batch_size):
        data = env.convert_to_ortools_input(index)
        optimal_values.append(solver(data, False))

    # Test the net
    # Set the agent to testing mode
    agent.training = False
    agent.stochastic_action_selection = False

    training_step = 0
    isDone = False

    episode_rewards = np.zeros((agent.batch_size, agent.num_items), dtype="float32")
    current_state, backpack_net_mask, item_net_mask, mha_used_mask = env.reset()
    dec_input = agent.generate_decoder_input(current_state)

    while not isDone:
        # Select an action
        backpack_id, item_id, decoded_item, backpack_net_mask = agent.act(
            current_state,
            dec_input,
            backpack_net_mask,
            item_net_mask,
            mha_used_mask,
            env.build_feasible_mask
        )

        # Play one step
        next_state, reward, isDone, info = env.step(
            backpack_id,
            item_id
        )
        
        # Store episode rewards
        episode_rewards[:, training_step] = reward[:, 0]

        # Update for next iteration
        dec_input = decoded_item
        current_state = next_state
        backpack_net_mask = info['backpack_net_mask']
        item_net_mask = info['item_net_mask']
        mha_used_mask = info['mha_used_mask']
        
        training_step += 1

    # print(episode_rewards)
    episode_rewards = np.sum(episode_rewards, axis=-1)
    
    stats = zip(optimal_values, episode_rewards)
    
    for s in stats:
        d_from_opt = 100 - (s[1] * 100 / s[0])
        print(f'Optimal {s[0]} | Net {s[1]} \t| Distance from Optimal {d_from_opt:.2f}')
    