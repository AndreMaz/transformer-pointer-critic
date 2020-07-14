import tensorflow as tf
# from models.transformer.utils import scaled_dot_product_attention


class PointerAttention(tf.keras.layers.Layer):
  def __init__(self, units = 64, out = 1):
    super(PointerAttention, self).__init__()
    # self.vocab_size = vocab_size

    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(out)

    # self.attention = Softmax(axis=1, name="attention")

  def call(self, dec_outputs, enc_outputs):
    
    # Unstack the input along time axis, i.e, axis = 1
    # Original data is a Tensor of  [ batch_size, time_steps, features] shape
    # After unstacking it is going to be a list (length = time_steps) of Tensors
    # Each element in list is going to be a Tensor of [ batch_size, features] shape
    steps = tf.unstack(dec_outputs, axis=1)
    pointerList = []

    # Iterate over time steps and compute the pointers
    for _, currentStep in enumerate(steps):
      # decoder_prev_hidden shape is [batch_size, features]
      # enc_output shape is [batch_size, timesteps, features]
      # To performs ops between them we need to reshape the decoder_prev_hidden into [batch_size, 1, features]
      decoder_prev_hidden_with_time_dim = tf.expand_dims(currentStep, 1)

      # score shape == (batch_size, max_length, 1)
      # we get 1 at the last axis because we are applying score to self.V
      # the shape of the tensor before applying self.V is (batch_size, max_length, units)
      score = self.V(tf.nn.tanh(
          self.W1(decoder_prev_hidden_with_time_dim) + self.W2(enc_outputs)))

      # Remove last dim
      score = tf.squeeze(score, axis=2)

      # Apply softmax
      attention_pointer = tf.nn.softmax(score, axis=-1)
      # attention_pointer = self.attention(score)
      
      # Store the pointer
      pointerList.append(attention_pointer)
    
    # Convert list back to tensor
    # Will create a time-major Tensor [time_steps, batch_size, features]
    pointerList = tf.convert_to_tensor(pointerList)

    # Put the data back into batch-major shape [batch_size, time_steps, features]
    return tf.transpose(pointerList, [1, 0, 2])

class PointerMultiHeadAttention(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads):
    super(PointerMultiHeadAttention, self).__init__()
    self.num_heads = num_heads
    self.d_model = d_model
    
    assert d_model % self.num_heads == 0
    
    self.depth = d_model // self.num_heads
    
    self.wq = tf.keras.layers.Dense(d_model)
    self.wk = tf.keras.layers.Dense(d_model)
    # self.wv = tf.keras.layers.Dense(d_model)
    
    # self.dense = tf.keras.layers.Dense(d_model)

    self.w_attention = tf.keras.layers.Dense(1, name='attention')

  def split_heads(self, x, batch_size):
    """Split the last (features) dimension into (num_heads, depth).
    Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
    """
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])
    
  def call(self, v, k, q, mask):
    batch_size = tf.shape(q)[0]
    
    q = self.wq(q)  # (batch_size, seq_len, d_model)
    k = self.wk(k)  # (batch_size, seq_len, d_model)
    # v = self.wv(v)  # (batch_size, seq_len, d_model)
    
    q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
    k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
    # v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
    
    # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
    # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
    combined_attention = scaled_dot_product_attention(
        q, k, v, mask, self.w_attention)
    
    return combined_attention

def scaled_dot_product_attention(q, k, v, mask, w_attention):
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

  # Reshape data into (batch, seq_len, seq_len, num_heads)
  reshaped_logits = tf.transpose(scaled_attention_logits, perm=[0, 2, 3, 1])

  reshaped_logits = w_attention(reshaped_logits) # (batch, seq_len, seq_len, 1)

  combined_attention_logits = tf.squeeze(reshaped_logits, -1) # (batch, seq_len, seq_len)
  combined_attention = tf.nn.softmax(combined_attention_logits, axis=-1)

  return combined_attention