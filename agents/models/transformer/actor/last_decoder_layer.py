import tensorflow as tf
from agents.models.transformer.common.attention import MultiHeadAttention
from agents.models.transformer.common.utils import point_wise_feed_forward_network

# from agents.transformer_pointer_critic.model.actor.custom_attention import PointerMultiHeadAttention
from agents.models.transformer.actor.pointer_attention import PointerAttention

class LastDecoderLayer(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, dff, rate=0.1):
    super(LastDecoderLayer, self).__init__()

    self.mha1 = MultiHeadAttention(d_model, num_heads)
    # self.mha2 = PointerMultiHeadAttention(d_model, num_heads)

    self.pointer_attention = PointerAttention(d_model)

    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    
    self.dropout1 = tf.keras.layers.Dropout(rate)
    
    self.BIG_NUMBER = 1e6

  def call(self,
           dec_input,
           enc_input,
           enc_outputs,
           training,
           mask,
           look_ahead_mask = None,
           padding_mask = None
           ):

    batch_size = enc_input.shape[0]
    # Create a tensor with the batch indices
    batch_indices = tf.convert_to_tensor(
            list(range(batch_size)), dtype='int32')

    # Encode the decoder input for the last time
    attn1, attn_weights_block1 = self.mha1(dec_input, dec_input, dec_input, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
    attn1 = self.dropout1(attn1, training=training)
    out1 = self.layernorm1(attn1 + dec_input)
    
    # Compute the pointers logits
    pointer_logits = self.pointer_attention(out1, enc_outputs)

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
