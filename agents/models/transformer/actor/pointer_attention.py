from tensorflow.keras.layers import Layer, Dense, Softmax
import tensorflow as tf
import numpy as np

class PointerAttention(Layer):
  def __init__(self, dense_units: int):
    super(PointerAttention, self).__init__()

    self.dense_units = dense_units

    self.W1 = Dense(self.dense_units)
    self.W2 = Dense(self.dense_units)
    self.V = Dense(1)

    self.BIG_NUMBER = 1e6

  def call(self,
           dec_output,
           enc_input,
           enc_outputs,
           mask,
           add_time_dim=False):
    batch_size = enc_input.shape[0]
    # Create a tensor with the batch indices
    batch_indices = tf.convert_to_tensor(
            list(range(batch_size)), dtype='int32')

    # To performs ops between them we need to reshape the decoder_prev_hidden into [batch_size, 1, features]
    if add_time_dim:
      decoder_prev_hidden_with_time_dim = tf.expand_dims(dec_output, 1)
    else:
      decoder_prev_hidden_with_time_dim = dec_output

    # score shape == [batch_size, max_length, 1]
    # we get 1 at the last axis because we are applying score to self.V
    # the shape of the tensor before applying self.V is [batch_size, max_length, units]
    score = self.V(tf.nn.tanh(
          self.W1(decoder_prev_hidden_with_time_dim) + self.W2(enc_outputs)))

    # Remove last dim
    pointer_logits = tf.squeeze(score, axis=2)

    # Apply the mask
    pointer_logits -= mask * self.BIG_NUMBER

        # Apply softmax
    pointer_probs = tf.nn.softmax(pointer_logits, axis=-1)

     # Grab the indice of the values pointed by the pointer
    pointer_index = pointer_probs.numpy().argmax(-1)[-1]

    # Grab decoded element
    dec_output = enc_input.numpy()[batch_indices, pointer_index]

    return pointer_logits,\
            pointer_probs,\
            pointer_index,\
            dec_output,
