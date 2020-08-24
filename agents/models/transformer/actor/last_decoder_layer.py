import tensorflow as tf
from agents.models.transformer.common.attention import MultiHeadAttention
from agents.models.transformer.common.utils import point_wise_feed_forward_network

# from agents.transformer_pointer_critic.model.actor.custom_attention import PointerMultiHeadAttention
from agents.models.transformer.actor.pointer_attention import PointerAttention

class LastDecoderLayer(tf.keras.layers.Layer):
  def __init__(self,
               d_model,
               num_heads,
               dff,
               attention_dense_units,
               rate=0.1):
    super(LastDecoderLayer, self).__init__()

    self.mha1 = MultiHeadAttention(d_model, num_heads)

    self.pointer_attention = PointerAttention(attention_dense_units)

    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    
    self.dropout1 = tf.keras.layers.Dropout(rate)
    
  def call(self,
           dec_input,
           enc_input,
           enc_output,
           training,
           attention_mask,
           look_ahead_mask = None,
           padding_mask = None
           ):

    # Encode the decoder input for the last time
    attn1, attn_weights_block1 = self.mha1(dec_input, dec_input, dec_input, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
    attn1 = self.dropout1(attn1, training=training)
    dec_output = self.layernorm1(attn1 + dec_input)
    
    # Compute the pointers logits, probs, position and the value
    p_logits, p_probs, p_index, p_value = self.pointer_attention(
        dec_output,
        enc_input,
        enc_output,
        attention_mask
    )

    return p_logits, p_probs, p_index, p_value
