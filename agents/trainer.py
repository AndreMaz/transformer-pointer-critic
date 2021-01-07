from agents.agent import Agent, TRANSFORMER
from environment.custom.knapsack.env_v2 import KnapsackV2
import tensorflow as tf
import numpy as np
import time

def trainer(env: KnapsackV2, agent: Agent, opts: dict, show_info: bool):
    # print(f'Training with {env.resource_sample_size} resources and {env.bin_sample_size} bins')

    training = True
    # General training vars
    n_iterations: int = opts['n_iterations']
    n_steps_to_update: int = opts['n_steps_to_update']
    average_rewards_buffer = []
    min_rewards_buffer = []
    max_rewards_buffer = []
    value_loss_buffer = []
    episode_count = 0
    
    # Initial vars for the initial episode
    isDone = False
    episode_rewards = np.zeros((agent.batch_size, agent.num_resources), dtype="float32")
    current_state, bin_net_mask, resource_net_mask, mha_used_mask = env.reset()
    dec_input = agent.generate_decoder_input(current_state)
    training_step = 0
    start = time.time()

    while episode_count < n_iterations:
        
        # Reached the end of episode. Reset for the next episode
        if isDone:
            isDone = False
            episode_rewards = np.zeros((agent.batch_size, agent.num_resources), dtype="float32")
            current_state, bin_net_mask, resource_net_mask, mha_used_mask = env.reset()
            # SOS input for resource Ptr Net
            dec_input = agent.generate_decoder_input(current_state)
            training_step = 0
            start = time.time()

        while not isDone or training_step > n_steps_to_update:
            # Select an action
            bin_id, resource_id, decoded_resource, bin_net_mask, _, _ = agent.act(
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

            # Store in memory
            agent.store(
                current_state,
                dec_input,
                bin_net_mask,
                resource_net_mask,
                mha_used_mask,
                resource_id,
                bin_id,
                reward,
                training_step
            )

            # Update for next iteration
            dec_input = decoded_resource
            current_state = next_state
            bin_net_mask = info['bin_net_mask']
            resource_net_mask = info['resource_net_mask']
            mha_used_mask = info['mha_used_mask']

            training_step += 1

            # print(f'CPU {np.average(current_state[:,1:env.bin_sample_size,0]):.5f} || RAM {np.average(current_state[:,1:env.bin_sample_size,1]):.5f} || MEM {np.average(current_state[:,1:env.bin_sample_size,2]):.5f}')

            # Grab the stats from the current episode
            if isDone:
                episode_count += 1

                average_per_problem = np.sum(episode_rewards, axis=-1)
                min_in_batch = np.min(average_per_problem, axis=-1)
                max_in_batch = np.max(average_per_problem, axis=-1)
                episode_reward = np.average(average_per_problem, axis=-1)
                
                average_rewards_buffer.append(episode_reward)
                min_rewards_buffer.append(min_in_batch)
                max_rewards_buffer.append(max_in_batch)
                break
                # current_state, bin_net_mask, resource_net_mask, mha_used_mask = env.reset()
        
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

        value_loss_buffer.append(value_loss)

        critic_grads = tape.gradient(
            value_loss,
            agent.critic.trainable_weights
        )

        with tf.GradientTape() as tape:
            resources_loss, decoded_resources = agent.compute_actor_loss(
                agent.resource_actor,
                agent.resource_masks,
                agent.resources,
                agent.decoded_resources,
                discounted_rewards,
                state_values
            )
        
        resource_grads = tape.gradient(
            resources_loss,
            agent.resource_actor.trainable_weights
        )

        # Add time dimension
        decoded_resources = tf.expand_dims(decoded_resources, axis=1)

        with tf.GradientTape() as tape:
            bin_loss, decoded_bins = agent.compute_actor_loss(
                agent.bin_actor,
                agent.bin_masks,
                agent.bins,
                decoded_resources,
                discounted_rewards,
                state_values
            )
        
        bin_grads = tape.gradient(
            bin_loss,
            agent.bin_actor.trainable_weights
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
                resource_grads,
                agent.resource_actor.trainable_weights
            )
        )

        agent.pointer_opt.apply_gradients(
            zip(
                bin_grads,
                agent.bin_actor.trainable_weights
            )
        )

        if isDone and show_info:
            print(f"\rEpisode: {episode_count} took {time.time() - start:.2f} seconds. Min in Batch: {min_in_batch:.3f} Max in Batch: {max_in_batch:.3f} Average Reward: {episode_reward:.3f} Value Loss: {value_loss[0]:.5f}", end="\n")

        # Iteration complete. Clear agent's memory
        agent.clear_memory()

    return average_rewards_buffer, min_rewards_buffer, max_rewards_buffer, value_loss_buffer
