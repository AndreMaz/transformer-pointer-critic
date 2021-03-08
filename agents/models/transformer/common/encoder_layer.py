import tensorflow as tf
from agents.models.transformer.common.attention import MultiHeadAttention
from agents.models.transformer.common.utils import point_wise_feed_forward_network

class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, dff, use_default_initializer: bool = True):
    super(EncoderLayer, self).__init__()

    self.mha = MultiHeadAttention(d_model, num_heads, use_default_initializer)
    self.ffn = point_wise_feed_forward_network(d_model, dff, use_default_initializer)

    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    
  def call(self, x, training, mask = None):

    attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
    out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)
    
    ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
    out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)
    
    return out2