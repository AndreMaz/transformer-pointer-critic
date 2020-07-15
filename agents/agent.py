# 
# from agents.transformer_pointer_critic.model.critic.model import CriticTransformer
# from agents.transformer_pointer_critic.model.actor.model import ActorTransformer

from agents.models.model_factory import model_factory

import tensorflow_probability as tfp
import tensorflow as tf
import numpy as np

TRANSFORMER = 'transformer'
LSTM = 'lstm'

class Agent():
    def __init__(self, name, opts):
        super(Agent, self).__init__()

        self.name = name

        # Default training mode
        self.training = True

        self.num_items = opts['num_items']
        self.num_backpacks = opts['num_backpacks']
        self.tensor_size = opts['tensor_size']
        self.vocab_size = opts['vocab_size']
        self.SOS_CODE = opts['actor']['SOS_CODE']

        self.gamma = opts['gamma'] # Discount factor
        self.entropy_coefficient = opts['entropy_coefficient']
        self.stochastic_action_selection = opts['stochastic_action_selection']

        ### Optimizers ###
        self.critic_learning_rate = opts['critic']['learning_rate']
        self.critic_opt = tf.keras.optimizers.Adam(learning_rate=self.critic_learning_rate)
        
        self.actor_learning_rate = opts['actor']['learning_rate']
        self.pointer_opt = tf.keras.optimizers.Adam(learning_rate=self.actor_learning_rate)

        ### Load the models
        self.item_actor, self.backpack_actor, self.critic = model_factory(
            name,
            opts
        )

        # Init memory
        self.states = []
        self.decoded_items = []
        self.items = []
        self.backpacks = []
        self.backpack_masks = []
        self.item_masks = []
        self.rewards = []
    
    def store(self,
              state,
              decoded_item,
              backpack_mask,
              items_masks,
              item,
              backpack,
              reward
            ):

        self.states.append(state[0]) # Remove batch dim
        
        self.decoded_items.append(decoded_item[0])

        self.backpack_masks.append(backpack_mask[0]) # Remove batch dim
        self.item_masks.append(items_masks[0]) # Remove batch dim

        self.items.append(item)
        self.backpacks.append(backpack)
        self.rewards.append(reward)

    def clear_memory(self):
        self.states = []
        self.decoded_items = []
        self.items = []
        self.backpack_masks = []
        self.item_masks = []
        self.backpacks = []
        self.rewards = []

    def generate_decoder_input(self, state):
        batch = state.shape[0]
        num_features = state.shape[2]
        dec_input = np.zeros((batch, 1, num_features), dtype='float32')

        dec_input += self.SOS_CODE

        return dec_input

    def compute_discounted_rewards(self, bootstrap_state_value):
        # print(bootstrap_state_value)
        bootstrap_state_value = bootstrap_state_value[0]

        # Discounted rewards
        discounted_rewards = []
        for reward in reversed(self.rewards):
            estimate_value = reward + self.gamma * bootstrap_state_value
            discounted_rewards.append(estimate_value)
            # Update bootstrap value for next iteration
            bootstrap_state_value = estimate_value

        # Reverse the discounted list
        discounted_rewards.reverse()
        discounted_rewards = tf.convert_to_tensor(discounted_rewards)
        discounted_rewards = tf.expand_dims(discounted_rewards, axis=1)

        return discounted_rewards

    def compute_value_loss(self, discounted_rewards):

        # Get state_values
        state_values = self.critic(
            tf.convert_to_tensor(self.states, dtype="float32"),
            self.training
        )

        value_loss = tf.keras.losses.mean_absolute_error(
            discounted_rewards,
            state_values
        )

        return value_loss, state_values

    def compute_actor_loss(self,
                           model,
                           masks,
                           actions,
                           decoder_inputs,
                           discounted_rewards,
                           state_values
                           ):
        
        advantages = discounted_rewards - state_values

        state = tf.convert_to_tensor(self.states, dtype='float32')
        mask = tf.convert_to_tensor(masks, dtype='float32')
        dec_input = tf.convert_to_tensor(decoder_inputs, dtype='float32')

        # Get logits, policy probabilities and actions
        pointer_logits, pointers_probs, point_index, dec_output = model(
            state,
            dec_input,
            mask,
            self.training
        )

        # One hot actions that we took during an episode
        actions_one_hot = tf.one_hot(
            actions, self.tensor_size, dtype="float32")

        # Compute the policy loss
        policy_loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=actions_one_hot, logits=pointer_logits)

        # Cross Entropy: Sum (Predicted_Prob_i * log(Predicted_Prob_i))
        # This improves exploration by discouraging premature converge to suboptimal deterministic policies
        # NOTE: Add small number to avoid doing log(0) = -Infinity
        entropy = tf.reduce_sum(
            pointers_probs * tf.math.log(pointers_probs + 1e-20), axis=1
        )

        policy_loss *= advantages
        policy_loss -= self.entropy_coefficient * entropy
        total_loss = tf.reduce_mean(policy_loss)

        return total_loss, dec_output
    
    train_step_signature = [
        tf.TensorSpec(shape=(None, 21, 2), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 21, 2), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 1), dtype=tf.float16),
        tf.TensorSpec(shape=(None, 1), dtype=tf.float16),
    ]

    # @tf.function(input_signature=train_step_signature)
    def act(self, state, dec_input, backpacks_mask, items_mask):
        #########################################
        ############ SELECT AN ITEM ############
        #########################################
        items_logits, items_probs, item_id, decoded_item = self.item_actor(
            state,
            dec_input,
            items_mask,
            self.training
        )

        if self.stochastic_action_selection:
            # Stochastic item selection
            dist_item = tfp.distributions.Categorical(probs = items_probs[0])
            item_id = dist_item.sample().numpy()
        else: 
            item_id = item_id[0]
        
        # Decode the item
        decoded_item = state[:, item_id]

        # Add batch dim
        decoded_item = tf.expand_dims(decoded_item, axis = 0)
        
        #########################################
        ### SELECT BACKPACK TO PLACE THE ITEM ###
        #########################################
        backpacks_logits, backpacks_probs, backpack_id, decoded_backpack = self.backpack_actor(
            state,
            decoded_item, # Pass decoded item to backpack decoder
            backpacks_mask,
            self.training
        )

        if self.stochastic_action_selection:
            # Stochastic backpack selection
            dist_backpack = tfp.distributions.Categorical(probs = backpacks_probs[0])
            backpack_id = dist_backpack.sample().numpy()
        else: 
            backpack_id = backpack_id[0]

        # Decode the backpack
        decoded_backpack = state[:, backpack_id]

        return backpack_id, \
               item_id, \
               decoded_item

    def set_training_mode(self, mode: bool):
        # Only used by transformer model
        if self.name == TRANSFORMER:
            self.item_actor.training = mode
            self.backpack_actor.training = mode
            self.critic = mode