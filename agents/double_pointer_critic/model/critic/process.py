from tensorflow.keras.layers import Layer, Dense, Softmax, LSTM
import tensorflow as tf
import numpy as np

### MORE INFO: https://github.com/MichelDeudon/neural-combinatorial-optimization-rl-tensorflow/blob/master/Ptr_Net_TSPTW/critic.py

class Process(Layer):
  def __init__(self,
               num_items: int,
               lstm_units: int,
               dense_units: int
              ):
    super(Process, self).__init__()
    
    self.num_items = num_items
    self.lstm_units = lstm_units
    self.dense_units = dense_units

    self.lstm = LSTM(
          self.lstm_units,
          return_sequences=True,
          return_state=True
      )

    self.W1 = Dense(self.dense_units)
    self.W2 = Dense(self.dense_units)
    self.V = Dense(1)

    # self.attention = Softmax(axis=1, name="attention")

  def call(self, enc_outputs, enc_hidden_state, enc_carry_state):

    # processing_input = enc_outputs
    batch_size = enc_outputs.shape[0]
    time_steps = enc_outputs.shape[1]
    features = enc_outputs.shape[2]
    processing_input = tf.zeros((batch_size, 1, features))

    for _ in range(time_steps):
      
      process_output, enc_hidden_state, enc_carry_state = self.lstm(
        processing_input,
        initial_state = [enc_hidden_state, enc_carry_state]
      )

      processing_input = self.single_step(enc_outputs, process_output)

    # Remove time dimension
    return tf.squeeze(processing_input, 1)
  
  def single_step(self, enc_outputs, processing_states):
    # Add time dimension. Will produce [batch, 1, features]
    # hidden_with_time_dim = tf.expand_dims(hidden, 1)

    # will produce [batch, features, 1]
    u = self.V(tf.nn.tanh(
          self.W1(processing_states) + self.W2(enc_outputs)))
    
    # u = tf.squeeze(u, 2)

    a = tf.nn.softmax(u, axis=1)

    a = tf.squeeze(a, 2)
    glimpse = tf.einsum('ij,ijk->ik', a, enc_outputs)

    # glimpse = tf.reduce_sum(a * enc_outputs, 1)

    glimpse = tf.expand_dims(glimpse, 1)


    return glimpse