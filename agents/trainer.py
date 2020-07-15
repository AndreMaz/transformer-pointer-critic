from agents.agent import Agent, TRANSFORMER
from environment.custom.knapsack.env import Knapsack
import tensorflow as tf


def trainer(env: Knapsack, agent: Agent, opts: dict):

    training = True
    n_iterations: int = opts['n_iterations']
    
    rewards_buffer = [0.0]
    for iteration in range(n_iterations):
        
        episode_reward = 0
        isDone = False
        current_state, backpack_net_mask, item_net_mask = env.reset()

        # SOS input for Item Ptr Net
        dec_input = agent.generate_decoder_input(current_state)

        while not isDone:
            # Select an action
            backpack_id, item_id, decoded_item = agent.act(
                current_state,
                dec_input,
                backpack_net_mask,
                item_net_mask
            )

            # Play one step
            next_state, reward, isDone, info = env.step(
                backpack_id,
                item_id
            )
            
            # Episode increment rewards
            episode_reward += reward

            # Store in memory
            agent.store(
                current_state,
                dec_input,
                backpack_net_mask,
                item_net_mask,
                item_id,
                backpack_id,
                reward
            )

            # Update for next iteration
            dec_input = decoded_item
            current_state = next_state
            backpack_net_mask = info['backpack_net_mask']
            item_net_mask = info['item_net_mask']

            # Prep the vars for the next training round
            if isDone:
                rewards_buffer.append(episode_reward)
                current_state, backpack_net_mask, item_net_mask = env.reset()
        
        print(f"\rIteration: {iteration}. Reward {episode_reward}", end="\n")

        if isDone == True:
            # We are done. So the state_value is 0
            bootstrap_state_value = tf.convert_to_tensor([0], dtype="float32")
        else:
            # Not done. Ask model to generate the state value
            backpack_id, item_id, bootstrap_state_value = agent.act(
                current_state)

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

        # Iteration complete. Clear agent's memory
        agent.clear_memory()

    return rewards_buffer
