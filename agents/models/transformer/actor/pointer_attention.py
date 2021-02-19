from tensorflow.keras.layers import Layer, Dense, Softmax
import tensorflow as tf
import numpy as np
from agents.models.transformer.common.utils import get_initializer

class PointerAttention(Layer):
  def __init__(self, dense_units: int, logit_clipping_C: float, use_default_initializer: bool = True):
    super(PointerAttention, self).__init__()

    self.dense_units = dense_units
    self.logit_clipping_C = logit_clipping_C
    self.use_default_initializer = use_default_initializer
    self.initializer = get_initializer(self.dense_units, self.use_default_initializer)

    self.W1 = Dense(self.dense_units, kernel_initializer=self.initializer)
    self.W2 = Dense(self.dense_units, kernel_initializer=self.initializer)
    self.V = Dense(1, kernel_initializer=self.initializer)

    self.BIG_NUMBER = 1e9

  def call(self,
           dec_output,
           enc_input,
           enc_outputs,
           mask,
           add_time_dim=False):
    batch_size = enc_input.shape[0]
    # Create a tensor with the batch indices
    batch_indices = tf.range(batch_size, dtype='int32')

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

    # Logits clipping
    # More info: https://arxiv.org/pdf/1611.09940.pdf
    # Appendix, Improving Exploration
    if self.logit_clipping_C is not None:
      pointer_logits = self.logit_clipping_C * tf.nn.tanh(pointer_logits)

    # Apply the mask
    pointer_logits -= mask * self.BIG_NUMBER

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

    return pointer_logits,\
            pointer_probs,\
            pointer_index,\
            dec_output,
