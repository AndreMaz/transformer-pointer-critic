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

        self.batch_size: int = opts['batch_size']
        self.num_items: int = opts['num_items']

        self.num_items: int = opts['num_items']
        self.num_backpacks: int = opts['num_backpacks']
        self.tensor_size: int = opts['tensor_size']
        self.vocab_size: int = opts['vocab_size']
        self.SOS_CODE: int = opts['actor']['SOS_CODE']

        self.gamma: float = opts['gamma'] # Discount factor
        self.entropy_coefficient: float = opts['entropy_coefficient']
        self.stochastic_action_selection: bool = opts['stochastic_action_selection']

        ### Optimizers ###
        self.critic_learning_rate: float = opts['critic']['learning_rate']
        self.critic_opt = tf.keras.optimizers.Adam(learning_rate=self.critic_learning_rate)
        
        self.actor_learning_rate: float = opts['actor']['learning_rate']
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
        self.mha_masks = [] # <= ToDo
        self.rewards = np.zeros((self.batch_size, self.num_items), dtype="float32")
    
    def store(self,
              state,
              decoded_item,
              backpack_mask,
              items_masks,
              mha_mask,
              item,
              backpack,
              reward,
              training_step
            ):

        self.states.append(state)
        
        self.decoded_items.append(decoded_item)

        self.backpack_masks.append(backpack_mask)
        self.item_masks.append(items_masks)
        self.mha_masks.append(mha_mask)

        self.items.append(item)
        self.backpacks.append(backpack)
        
        self.rewards[:, training_step] = reward[:, 0]

    def clear_memory(self):
        self.states = []
        self.decoded_items = []
        
        self.backpack_masks = []
        self.item_masks = []
        self.mha_masks = []

        self.items = []
        self.backpacks = []

        self.rewards = np.zeros((self.batch_size, self.num_items), dtype="float32")

    def generate_decoder_input(self, state):
        batch = state.shape[0]
        num_features = state.shape[2]
        dec_input = np.zeros((batch, 1, num_features), dtype='float32')

        dec_input += self.SOS_CODE

        return dec_input

    def compute_discounted_rewards(self, bootstrap_state_value):
        bootstrap_state_value = tf.reshape(bootstrap_state_value, [bootstrap_state_value.shape[0]])

        # self.rewards = np.ones((2, 10), dtype='float32')

        # Discounted rewards
        discounted_rewards = np.zeros_like(self.rewards)
        for step in reversed(range(self.rewards.shape[1])):
            estimate_value = self.rewards[:, step] + self.gamma * bootstrap_state_value
            discounted_rewards[:, step] = estimate_value

            # Update bootstrap value for next iteration
            bootstrap_state_value = estimate_value

        return discounted_rewards

    def compute_value_loss(self, discounted_rewards):
        # Reshape data into a single forward pass format
        states = tf.convert_to_tensor(self.states, dtype="float32")
        batch, dec_steps, elem, features = states.shape
        states = tf.reshape(states, [batch*dec_steps, elem, features])

        mha_mask = tf.convert_to_tensor(self.mha_masks, dtype='float32')
        mha_mask = tf.reshape(mha_mask, [batch*dec_steps, 1, 1, elem])        

        # Get state_values
        state_values = self.critic(
            states,
            self.training,
            enc_padding_mask = mha_mask
        )

        # Reshape the rewards bach into [batch, dec_step] form
        state_values = tf.reshape(state_values, [batch, dec_steps])
        state_values = tf.transpose(state_values, [1, 0])

        value_loss = tf.keras.losses.mean_absolute_error(
            discounted_rewards,
            state_values
        )

        value_loss = value_loss / self.batch_size

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

        states = tf.convert_to_tensor(self.states, dtype='float32')
        batch, dec_steps, elem, features = states.shape
        states = tf.reshape(states, [batch*dec_steps, elem, features])

        attention_mask = tf.convert_to_tensor(masks, dtype='float32')
        attention_mask = tf.reshape(attention_mask, [batch*dec_steps, elem])

        mha_mask = tf.convert_to_tensor(self.mha_masks, dtype='float32')
        mha_mask = tf.reshape(mha_mask, [batch*dec_steps, 1, 1, elem])
        
        dec_input = tf.convert_to_tensor(decoder_inputs, dtype='float32')
        dec_input = tf.reshape(dec_input, [batch*dec_steps, 1, features])

        # Get logits, policy probabilities and actions
        pointer_logits, pointers_probs, point_index, dec_output = model(
            states,
            dec_input,
            attention_mask,
            self.training,
            enc_padding_mask = mha_mask,
            dec_padding_mask = mha_mask
        )

        # One hot actions that we took during an episode
        actions_one_hot = tf.one_hot(
            actions, self.tensor_size, dtype="float32")

        actions_one_hot = tf.reshape(actions_one_hot, [batch*dec_steps, elem])

        # Compute the policy loss
        policy_loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=actions_one_hot, logits=pointer_logits)

        policy_loss = tf.reshape(policy_loss, [batch, dec_steps])
        policy_loss = tf.transpose(policy_loss, [1, 0])

        # Cross Entropy: Sum (Predicted_Prob_i * log(Predicted_Prob_i))
        # This improves exploration by discouraging premature converge to suboptimal deterministic policies
        # NOTE: Add small number to avoid doing log(0) = -Infinity
        entropy = tf.reduce_sum(
            pointers_probs * tf.math.log(pointers_probs + 1e-20), axis=1
        )

        entropy = tf.reshape(entropy, [batch, dec_steps])
        entropy = tf.transpose(entropy, [1, 0])

        policy_loss *= advantages
        policy_loss -= self.entropy_coefficient * entropy
        total_loss = tf.reduce_mean(policy_loss)

        total_loss = total_loss / self.batch_size

        return total_loss, dec_output

    def act(self,
            state,
            dec_input,
            backpacks_mask,
            items_mask,
            mha_used_mask,
            build_feasible_mask
        ):
        batch_size = state.shape[0]
        # Create a tensor with the batch indices
        batch_indices = tf.range(batch_size, dtype='int32')

        #########################################
        ############ SELECT AN ITEM ############
        #########################################
        items_logits, items_probs, item_ids, decoded_items = self.item_actor(
            state,
            dec_input,
            items_mask,
            self.training,
            enc_padding_mask = mha_used_mask,
            dec_padding_mask = mha_used_mask
        )

        if self.stochastic_action_selection:
            item_ids = []
            for batch_id in range(batch_size):
                # Stochastic item selection
                dist_item = tfp.distributions.Categorical(probs = items_probs[batch_id])
                # Sample from distribution
                item_ids.append(dist_item.sample().numpy())
        
        # Decode the items
        decoded_items = state[batch_indices, item_ids]

        # Update the masks for the backpack
        # This will only allow to point to feasible solutions
        backpacks_mask = build_feasible_mask(state,
                                             decoded_items,
                                             backpacks_mask
                                             )

        # Add time step dim
        decoded_items = tf.expand_dims(decoded_items, axis = 1)
        
        #########################################
        ### SELECT BACKPACK TO PLACE THE ITEM ###
        #########################################
        backpacks_logits, backpacks_probs, backpack_ids, decoded_backpack = self.backpack_actor(
            state,
            decoded_items, # Pass decoded item to backpack decoder
            backpacks_mask,
            self.training,
            enc_padding_mask = mha_used_mask,
            dec_padding_mask = mha_used_mask
        )

        if self.stochastic_action_selection:
            backpack_ids = []
            for batch_id in range(batch_size):
                # Stochastic backpack selection
                dist_backpack = tfp.distributions.Categorical(probs = backpacks_probs[batch_id])
                backpack_ids.append(dist_backpack.sample().numpy())

        # Decode the backpack
        decoded_backpack = state[batch_indices, backpack_ids]

        #for index, bp_id in enumerate(backpack_ids):
        #    if (bp_id < 32):
        #        print(bp_id)
        #        print(backpacks_mask[index])

        return backpack_ids, \
               item_ids, \
               decoded_items, \
               backpacks_mask

    def set_training_mode(self, mode: bool):
        # Only used by transformer model
        if self.name == TRANSFORMER:
            self.item_actor.training = mode
            self.backpack_actor.training = mode
            self.critic = mode
