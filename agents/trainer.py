from agents.agent import Agent, TRANSFORMER
from environment.custom.knapsack.env_v2 import KnapsackV2
import tensorflow as tf
import numpy as np
import time

def trainer(env: KnapsackV2, agent: Agent, opts: dict):
    print(f'Training with {env.item_sample_size} items and {env.backpack_sample_size} backpacks')

    training = True
    # General training vars
    n_iterations: int = opts['n_iterations']
    n_steps_to_update: int = opts['n_steps_to_update']
    rewards_buffer = []
    episode_count = 0
    
    # Initial vars for the initial episode
    isDone = False
    episode_rewards = np.zeros((agent.batch_size, agent.num_items), dtype="float32")
    current_state, backpack_net_mask, item_net_mask, mha_used_mask = env.reset()
    dec_input = agent.generate_decoder_input(current_state)
    training_step = 0
    start = time.time()

    while episode_count < n_iterations:
        
        # Reached the end of episode. Reset for the next episode
        if isDone:
            isDone = False
            episode_rewards = np.zeros((agent.batch_size, agent.num_items), dtype="float32")
            current_state, backpack_net_mask, item_net_mask, mha_used_mask = env.reset()
            # SOS input for Item Ptr Net
            dec_input = agent.generate_decoder_input(current_state)
            training_step = 0
            start = time.time()

        while not isDone or training_step > n_steps_to_update:
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

            # Store in memory
            agent.store(
                current_state,
                dec_input,
                backpack_net_mask,
                item_net_mask,
                mha_used_mask,
                item_id,
                backpack_id,
                reward,
                training_step
            )

            # Update for next iteration
            dec_input = decoded_item
            current_state = next_state
            backpack_net_mask = info['backpack_net_mask']
            item_net_mask = info['item_net_mask']
            mha_used_mask = info['mha_used_mask']

            training_step += 1

            # Grab the stats from the current episode
            if isDone:
                episode_count += 1

                average_per_problem = np.sum(episode_rewards, axis=-1)
                episode_reward = np.average(average_per_problem, axis=-1)
                rewards_buffer.append(episode_reward)
                # current_state, backpack_net_mask, item_net_mask, mha_used_mask = env.reset()
        
        if isDone == True:
            # We are done. So the state_value is 0
            bootstrap_state_value = tf.zeros([agent.batch_size, 1],dtype="float32")
        else:
            # Not done. Ask model to generate the state value
            bootstrap_state_value = agent.critic(current_state, agent.training)

        discounted_rewards = agent.compute_discounted_rewards(bootstrap_state_value)
        
        ### Update Critic ###
        with tf.GradientTape() as tape:
            value_loss, state_values = agent.compute_value_loss(
                discounted_rewards
            )

        critic_grads = tape.gradient(
            value_loss,
            agent.critic.trainable_weights
        )

        with tf.GradientTape() as tape:
            items_loss, decoded_items = agent.compute_actor_loss(
                agent.item_actor,
                agent.item_masks,
                agent.items,
                agent.decoded_items,
                discounted_rewards,
                state_values
            )
        
        item_grads = tape.gradient(
            items_loss,
            agent.item_actor.trainable_weights
        )

        # Add time dimension
        decoded_items = tf.expand_dims(decoded_items, axis=1)

        with tf.GradientTape() as tape:
            backpack_loss, decoded_backpacks = agent.compute_actor_loss(
                agent.backpack_actor,
                agent.backpack_masks,
                agent.backpacks,
                decoded_items,
                discounted_rewards,
                state_values
            )
        
        backpack_grads = tape.gradient(
            backpack_loss,
            agent.backpack_actor.trainable_weights
        )

        # Apply gradients to tweak the model
        agent.critic_opt.apply_gradients(
            zip(
                critic_grads,
                agent.critic.trainable_weights
            )
        )

        agent.pointer_opt.apply_gradients(
            zip(
                item_grads,
                agent.item_actor.trainable_weights
            )
        )

        agent.pointer_opt.apply_gradients(
            zip(
                backpack_grads,
                agent.backpack_actor.trainable_weights
            )
        )

        if isDone:
            print(f"\rEpisode: {episode_count} took {time.time() - start:.2f} seconds. Average Reward: {episode_reward:.3f}", end="\n")

        # Iteration complete. Clear agent's memory
        agent.clear_memory()

    return rewards_buffer
