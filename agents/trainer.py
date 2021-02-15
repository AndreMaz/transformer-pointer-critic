from agents.agent import Agent, TRANSFORMER
from environment.custom.knapsack.env_v2 import KnapsackV2
import tensorflow as tf
import numpy as np
import time

def trainer(env: KnapsackV2, agent: Agent, opts: dict, show_progress: bool):
    # print(f'Training with {env.resource_sample_size} resources and {env.bin_sample_size} bins')

    training = True

    single_actor: bool = agent.single_actor
    # General training vars
    n_iterations: int = opts['n_iterations']
    n_steps_to_update: int = opts['n_steps_to_update']
    update_resource_decoder_input: bool = opts['update_resource_decoder_input']

    average_rewards_buffer = []
    min_rewards_buffer = []
    max_rewards_buffer = []
    value_loss_buffer = []

    resources_policy_loss_buffer = []
    resources_total_loss_buffer = []
    resources_entropy_buffer = []
    
    bins_policy_loss_buffer = []
    bins_total_loss_buffer = []
    bins_entropy_buffer = []

    episode_count = 0
    
    # Initial vars for the initial episode
    isDone = False
    episode_rewards = np.zeros((agent.batch_size, agent.num_resources), dtype="float32")
    if single_actor:
        current_state, dec_input, bin_net_mask, mha_used_mask = env.reset()
        resource_net_mask = np.array(None)
    else:
        current_state, bin_net_mask, resource_net_mask, mha_used_mask = env.reset()
        dec_input = agent.generate_decoder_input(current_state)
    training_step = 0
    start = time.time()

    while episode_count < n_iterations:
        
        # Reached the end of episode. Reset for the next episode
        if isDone:
            isDone = False
            episode_rewards = np.zeros((agent.batch_size, agent.num_resources), dtype="float32")
            if single_actor:
                current_state, dec_input, bin_net_mask, mha_used_mask = env.reset()
                resource_net_mask = np.array(None)
            else:
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

            if single_actor:
                next_state, decoded_resource, reward, isDone, info = env.step(
                    bin_id,
                    resource_id,
                    bin_net_mask
                )
            else:
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
                current_state.copy(),
                dec_input.copy(),
                bin_net_mask.copy(),
                resource_net_mask.copy(),
                mha_used_mask.copy(),
                resource_id.copy(),
                bin_id.numpy().copy(),
                reward.numpy().copy(),
                training_step
            )

            # Update for next iteration
            if single_actor:
                dec_input = decoded_resource.copy()
            else:
                if update_resource_decoder_input:
                    dec_input = decoded_resource.numpy()

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
            bootstrap_state_value = agent.critic(
                current_state, agent.training, enc_padding_mask = mha_used_mask
            )

        discounted_rewards = agent.compute_discounted_rewards(bootstrap_state_value)
        
        ### Update Critic ###
        with tf.GradientTape() as tape_value:
            value_loss, state_values, advantages = agent.compute_value_loss(
                discounted_rewards
            )

        critic_grads = tape_value.gradient(
            value_loss,
            agent.critic.trainable_weights
        )

        if not single_actor:
            with tf.GradientTape() as tape_resource:
                resources_loss,\
                    decoded_resources,\
                    resources_entropy,\
                    resource_policy_loss  = agent.compute_actor_loss(
                    agent.resource_actor,
                    agent.resource_masks,
                    agent.resources,
                    agent.resource_net_decoder_input,
                    tf.stop_gradient(advantages)
                )
            
            resource_grads = tape_resource.gradient(
                resources_loss,
                agent.resource_actor.trainable_weights
            )

            # Add time dimension
            decoded_resources = tf.expand_dims(decoded_resources, axis=1)
        else:
            resource_policy_loss = np.array([0], dtype="float32")
            resources_entropy = np.array([0], dtype="float32")
            resources_loss = tf.constant(0, dtype="float32")
            
            decoded_resources = agent.resource_net_decoder_input
        
        with tf.GradientTape() as tape_bin:
            bin_loss,\
                decoded_bins,\
                bin_entropy,\
                bin_policy_loss = agent.compute_actor_loss(
                agent.bin_actor,
                agent.bin_masks,
                agent.bins,
                decoded_resources,
                tf.stop_gradient(advantages)
            )
        
        bin_grads = tape_bin.gradient(
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

        if not single_actor:
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

        # Store the stats
        value_loss_buffer.append(value_loss.numpy())
        resources_policy_loss_buffer.append(np.mean(resource_policy_loss))
        resources_total_loss_buffer.append(resources_loss.numpy())
        resources_entropy_buffer.append(np.mean(resources_entropy))
        bins_policy_loss_buffer.append(np.mean(bin_policy_loss))
        bins_total_loss_buffer.append(bin_loss.numpy())
        bins_entropy_buffer.append(np.mean(bin_entropy))

        if isDone and show_progress:
            print(f"\rEp: {episode_count} took {time.time() - start:.2f} sec.\t" +
            f"Min@Batch: {min_in_batch:.3f}\t" +
            f"Max@Batch: {max_in_batch:.3f}\t" +
            f"Avg@Batch: {episode_reward:.3f}\t" +
            f"V_Loss: {value_loss:.3f}\t" +
            f"R_Tot_Loss: {resources_loss:.3f}\t" +
            f"R_Pol_Loss: {tf.reduce_mean(resource_policy_loss):.3f}\t" +
            f"Entr R: {tf.reduce_mean(resources_entropy):.3f}\t" +
            f"B_Tot_Loss: {bin_loss:.3f}\t" + 
            f"B_Pol_Loss: {tf.reduce_mean(bin_policy_loss):.3f}\t" +
            f"Entr B: {tf.reduce_mean(bin_entropy):.3f}", end="\n")

        # Iteration complete. Clear agent's memory
        agent.clear_memory()

    return average_rewards_buffer,\
        min_rewards_buffer,\
        max_rewards_buffer,\
        value_loss_buffer, \
        resources_policy_loss_buffer,\
        resources_total_loss_buffer,\
        resources_entropy_buffer,\
        bins_policy_loss_buffer,\
        bins_total_loss_buffer,\
        bins_entropy_buffer
