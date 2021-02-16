# 
# from agents.transformer_pointer_critic.model.critic.model import CriticTransformer
# from agents.transformer_pointer_critic.model.actor.model import ActorTransformer

from agents.models.model_factory import model_factory
from environment.custom.resource_v3.utils import reshape_into_vertical_format, reshape_into_horizontal_format

import tensorflow_probability as tfp
import tensorflow as tf
import numpy as np

TRANSFORMER = 'transformer'
LSTM = 'lstm'

class Agent():
    def __init__(self, name, opts):
        super(Agent, self).__init__()

        self.name = name
        self.agent_config = opts

        # Default training mode
        self.training = True

        self.single_actor: bool = opts['single_actor']

        if self.single_actor:
            assert self.single_actor and opts['generate_decoder_input'], "For using single actor set generate_decoder_input in env. config to True"

        self.batch_size: int = opts['batch_size']
        self.num_resources: int = opts['num_resources']

        self.num_bins: int = opts['num_bins']
        self.tensor_size: int = opts['tensor_size']
        self.vocab_size: int = opts['vocab_size']
        self.SOS_CODE: int = opts['actor']['SOS_CODE']

        self.gamma: float = opts['gamma'] # Discount factor
        self.values_loss_coefficient: float = opts['values_loss_coefficient']
        self.entropy_coefficient: float = opts['entropy_coefficient']
        self.stochastic_action_selection: bool = opts['stochastic_action_selection']

        ### Optimizers ###
        self.critic_learning_rate: float = opts['critic']['learning_rate']
        self.critic_opt = tf.keras.optimizers.Adam(
            learning_rate=self.critic_learning_rate,
            #clipnorm=1.0
        )
        
        self.actor_learning_rate: float = opts['actor']['learning_rate']
        self.pointer_opt = tf.keras.optimizers.Adam(
            learning_rate=self.actor_learning_rate,
            #clipnorm=1.0
        )

        # Error fn for the critic
        self.mse = tf.keras.losses.MeanSquaredError()

        # Error fn for the actor
        self.loss_fn = tf.keras.losses.CategoricalCrossentropy(
            from_logits=True
        )

        ### Load the models
        self.resource_actor, self.bin_actor, self.critic = model_factory(
            name,
            opts
        )

        # Init memory
        self.states = []
        self.resource_net_decoder_input = []
        self.bin_net_decoder_input = []
        self.resources = []
        self.bins = []
        self.bin_masks = []
        self.resource_masks = []
        self.mha_masks = [] # <= ToDo
        self.rewards = np.zeros((self.batch_size, self.num_resources), dtype="float32")

    def store(self,
              state,
              resource_net_dec_input,
              bin_net_dec_input,
              bin_mask,
              resources_masks,
              mha_mask,
              resource,
              bin,
              reward,
              training_step
            ):

        self.states.append(state)
        
        self.resource_net_decoder_input.append(resource_net_dec_input)
        self.bin_net_decoder_input.append(bin_net_dec_input)

        self.bin_masks.append(bin_mask)
        self.resource_masks.append(resources_masks)
        self.mha_masks.append(mha_mask)

        self.resources.append(resource) # Resource IDs, a.k.a, resource actions
        self.bins.append(bin) # Bind IDs, a.k.a, bin actions
        
        self.rewards[:, training_step] = reward[:, 0]

    def clear_memory(self):
        self.states = []

        self.resource_net_decoder_input = []
        self.bin_net_decoder_input = []

        self.bin_masks = []
        self.resource_masks = []
        self.mha_masks = []

        self.resources = []
        self.bins = []

        self.rewards = np.zeros((self.batch_size, self.num_resources), dtype="float32")

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
        og_batch_size = self.states[0].shape[0]

        # Reshape data into a single forward pass format
        # shape=(batch_steps * num_resources, total_elems, features)
        states = tf.concat(self.states, axis=0)
        batch_size = states.shape[0]

        # shape=(batch_steps * num_resources, 1, 1, total_elems)
        mha_mask = tf.concat(self.mha_masks, axis=0)

        # Get state_values
        state_values = self.critic(
            states,
            self.training,
            enc_padding_mask = mha_mask
        )

        #a = np.array([
        #    [1, 2],
        #    [3, 4]
        # ], dtype="float32")

        # Reshape the discounted rewards to match state values
        discounted_rewards = reshape_into_vertical_format(
            discounted_rewards, batch_size
        )

        advantages = discounted_rewards - state_values

        # Compute average loss for the batch
        # value_loss = self.mse(discounted_rewards, state_values)
        value_loss = tf.keras.losses.mean_squared_error(discounted_rewards, state_values)

        # Apply a constant factor
        value_loss = self.values_loss_coefficient * value_loss

        value_loss = tf.reduce_mean(value_loss, axis=-1)

        return value_loss, state_values, advantages
    
    def compute_actor_loss(self,
                           model,
                           masks,
                           actions,
                           decoder_inputs,
                           advantages
                           ):

        states = tf.concat(self.states, axis=0)
        attention_mask = tf.concat(masks, axis=0)
        mha_mask = tf.concat(self.mha_masks, axis=0)
        dec_input = tf.concat(decoder_inputs, axis=0)

        # Get logits, policy probabilities and actions
        pointer_logits, pointers_probs, point_index, dec_output = model(
            states,
            dec_input,
            attention_mask,
            self.training,
            enc_padding_mask = mha_mask,
            dec_padding_mask = mha_mask
        )

        pointers_probs = tf.stop_gradient(pointers_probs)
        point_index = tf.stop_gradient(point_index)
        dec_output = tf.stop_gradient(dec_output)

        # One hot actions that we took during an episode
        actions = tf.concat(actions, axis=0)
        # actions_one_hot = tf.one_hot(
        #    actions, self.tensor_size, dtype="float32")

        # Compute the policy loss
        policy_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.NONE
        )
        policy_loss = policy_loss_fn(
            actions, pointer_logits
        )

        # # # # # # # # # # # # # # 
        # for debugging purposes #
        # # # # # # # # # # # # # # 
        # mean_allowed = tf.reduce_mean(policy_loss)
        # if mean_allowed > 10:
        #     max_index = tf.argmax(policy_loss)
        #     print(max_index)
        #     print(policy_loss[max_index])
        #     print(actions[max_index])
        #     print(attention_mask[max_index])
        #     print(pointers_probs[max_index])
        #     print(pointer_logits[max_index])
        #     assert False, "EXIT"

        policy_loss_times_adv = policy_loss * tf.squeeze(advantages,axis=-1)

        # Entropy loss can be calculated as cross-entropy over itself.
        # The greater the entropy, the more random the actions an agent takes.
        entropy_fn = tf.keras.losses.CategoricalCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.NONE
        )
        entropy = entropy_fn(pointers_probs, pointer_logits)

        # entropy = tf.nn.softmax_cross_entropy_with_logits(
        #    labels=tf.stop_gradient(pointers_probs), logits=pointer_logits)

        # Compute average entropy loss
        # entropy_loss = tf.reduce_mean(entropy_loss)
        total_loss = tf.reduce_mean(
        policy_loss_times_adv - self.entropy_coefficient * entropy, axis=-1
            )
        

        return total_loss, dec_output, entropy, policy_loss

    def act(self,
            state,
            dec_input,
            bins_mask,
            resources_mask,
            mha_used_mask,
            build_feasible_mask
        ):
        batch_size = state.shape[0]
        # Create a tensor with the batch indices
        batch_indices = tf.range(batch_size, dtype='int32')

        if not self.single_actor:
            #########################################
            ############ SELECT AN resource ############
            #########################################
            resources_logits, resources_probs, resource_ids, decoded_resources = self.resource_actor(
                state,
                dec_input,
                resources_mask,
                self.training,
                enc_padding_mask = mha_used_mask,
                dec_padding_mask = mha_used_mask
            )

            if self.stochastic_action_selection:
                # resource_ids = []
                # for batch_id in range(batch_size):
                # Stochastic resource selection
                dist_resource = tfp.distributions.Categorical(probs = resources_probs)
                # Sample from distribution
                resource_ids = dist_resource.sample()
            
            resource_ids = resource_ids.numpy()
            # Decode the resources
            decoded_resources = state[batch_indices, resource_ids]
            
            # Add time step dim
            decoded_resources = tf.expand_dims(decoded_resources, axis = 1).numpy()
        else:
            resource_ids = np.array(None)
            resources_probs = None
            # Resource are provided by the environment
            decoded_resources = dec_input.copy()

        # Update the masks for the bin
        # This will only allow to point to feasible solutions
        bins_mask = build_feasible_mask(state,
                                            decoded_resources,
                                            bins_mask
                                            )

        #########################################
        ### SELECT bin TO PLACE THE resource ###
        #########################################
        bins_logits, bins_probs, bin_ids, decoded_bin = self.bin_actor(
            state,
            decoded_resources, # Pass decoded resource to bin decoder
            bins_mask,
            self.training,
            enc_padding_mask = mha_used_mask,
            dec_padding_mask = mha_used_mask
        )

        if self.stochastic_action_selection:
            # bin_ids = []
            # for batch_id in range(batch_size):
            # Stochastic bin selection
            dist_bin = tfp.distributions.Categorical(probs = bins_probs)
            bin_ids = dist_bin.sample()

        # Decode the bin
        decoded_bin = state[batch_indices, bin_ids]

        return bin_ids, \
               resource_ids, \
               decoded_resources, \
               bins_mask, \
               resources_probs, \
               bins_probs

    def set_training_mode(self, mode: bool):
        # Only used by transformer model
        if self.name == TRANSFORMER:
            self.resource_actor.training = mode
            self.bin_actor.training = mode
            self.critic = mode

    def save_weights(self, location):

        self.resource_actor.save_weights(f'{location}/resource_actor')
        self.bin_actor.save_weights(f'{location}/bin_actor')


    def load_weights(self, location):

        self.resource_actor.load_weights(f'{location}/resource_actor')
        self.bin_actor.load_weights(f'{location}/bin_actor')

    def set_testing_mode(self, batch_size, num_resources):
        self.training = False
        self.stochastic_action_selection = False

        self.batch_size = batch_size
        # Set the number for resources during testing
        # i.e, number of steps
        self.num_resources = num_resources
    
def process_logits(pointer_logits, enc_input):
    batch_size = enc_input.shape[0]
    # Create a tensor with the batch indices
    batch_indices = tf.range(batch_size, dtype='int32')

    # Apply softmax
    pointer_probs = tf.nn.softmax(pointer_logits, axis=-1)

    # Grab the indice of the values pointed by the pointer
    # pointer_index = pointer_probs.numpy().argmax(-1)
    pointer_index = tf.argmax(pointer_probs, axis=-1, output_type='int32')

    # Grab decoded element
    # dec_output = enc_input.numpy()[batch_indices, pointer_index]
    dec_output = tf.gather_nd(
        enc_input, tf.stack((batch_indices, pointer_index), -1)
    )

    return pointer_probs, pointer_index, dec_output