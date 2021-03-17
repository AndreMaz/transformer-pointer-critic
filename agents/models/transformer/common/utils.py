import numpy as np
import tensorflow as tf
import math

def point_wise_feed_forward_network(d_model, dff, use_default_initializer: bool = True):
  initializer = get_initializer(d_model, use_default_initializer)

  return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu', kernel_initializer=initializer),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model, kernel_initializer=initializer)  # (batch_size, seq_len, d_model)
  ])

def scaled_dot_product_attention(q, k, v, mask):
  """Calculate the attention weights.
  q, k, v must have matching leading dimensions.
  k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
  The mask has different shapes depending on its type(padding or look ahead) 
  but it must be broadcastable for addition.
  
  Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable 
          to (..., seq_len_q, seq_len_k). Defaults to None.
    
  Returns:
    output, attention_weights
  """

  matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
  
  # scale matmul_qk
  dk = tf.cast(tf.shape(k)[-1], tf.float32)
  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

  # add the mask to the scaled tensor.
  if mask is not None:
    scaled_attention_logits += (mask * -1e9)  

  # softmax is normalized on the last axis (seq_len_k) so that the scores
  # add up to 1.
  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

  output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

  return output, attention_weights

def get_initializer(dims: int, use_default_initializer: bool):
    if use_default_initializer: 
      # Default initializer. More info: https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense
      return 'glorot_uniform'

    # From https://arxiv.org/pdf/1803.08475.pdf
    # Page 6. Section 5 Hyperparameters
    value = 1 / math.sqrt(dims)

    init = tf.keras.initializers.RandomUniform(
        minval=-1*value,
        maxval=value,
        seed=None
    )

    return init

