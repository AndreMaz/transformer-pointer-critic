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

  def call(self, dec_output, enc_outputs):
    
    # decoder_prev_hidden shape is [batch_size, features]
    # enc_output shape is [batch_size, timesteps, features]
    # To performs ops between them we need to reshape the decoder_prev_hidden into [batch_size, 1, features]
    
    # decoder_prev_hidden_with_time_dim = tf.expand_dims(dec_output, 1)
    decoder_prev_hidden_with_time_dim = dec_output

    # score shape == [batch_size, max_length, 1]
    # we get 1 at the last axis because we are applying score to self.V
    # the shape of the tensor before applying self.V is [batch_size, max_length, units]
    score = self.V(tf.nn.tanh(
          self.W1(decoder_prev_hidden_with_time_dim) + self.W2(enc_outputs)))

    # Remove last dim
    pointer_logits = tf.squeeze(score, axis=2)

    return pointer_logits