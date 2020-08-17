import imp

from tensorflow.python.ops.gen_data_flow_ops import PriorityQueue
from agents.optimum_solver import solver
import numpy as np

def test(env, agent):
    data = env.convert_to_ortools_input()
    solver(data, False)
    
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


    average_per_problem = np.sum(episode_rewards, axis=-1)
    episode_reward = np.average(average_per_problem, axis=-1)
    
    print(f'Average Reward: {episode_reward}')
    print(f'Detailed View: {average_per_problem}')