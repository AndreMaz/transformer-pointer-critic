import tensorflow as tf
from tensorflow.keras.layers import TimeDistributed, Dense
import numpy as np

from agents.transformer_pointer_critic.model.actor.decoder_layer import DecoderLayer
from agents.transformer_pointer_critic.model.common.utils import positional_encoding

from agents.transformer_pointer_critic.model.actor.last_decoder_layer import LastDecoderLayer

class Decoder(tf.keras.layers.Layer):
  def __init__(self,
               num_layers,
               d_model,
               num_heads,
               dff,
               SOS_CODE,
               target_vocab_size,
               maximum_position_encoding,
               embedding_time_distributed: bool,
               rate=0.1):
    super(Decoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers
    self.embedding_time_distributed = embedding_time_distributed
    self.vocab_size = target_vocab_size

    self.SOS_CODE = SOS_CODE

    self.d_model = d_model
    self.num_layers = num_layers
    
    # self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
    if self.embedding_time_distributed:
      self.embedding = TimeDistributed(
          Dense(
              self.d_model
          )
      )
    else:
      self.embedding = Dense(
          self.d_model
      )


    self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)
    
    self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) 
                       for _ in range(num_layers)]

    self.last_decoder_layer = LastDecoderLayer(d_model, num_heads, dff, rate)

    self.dropout = tf.keras.layers.Dropout(rate)
    
  def call(self,
           dec_input,
           enc_input,
           enc_output,
           training,
           attention_mask,
           look_ahead_mask,
           padding_mask):

    seq_len = tf.shape(dec_input)[1]
    attention_weights = {}
    
    dec_input = self.embedding(dec_input)  # (batch_size, target_seq_len, d_model)
    dec_input *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    dec_input += self.pos_encoding[:, :seq_len, :]
    
    dec_input = self.dropout(dec_input, training=training)

    for i in range(self.num_layers):
      dec_input, block1, block2 = self.dec_layers[i](dec_input,
                                                     enc_output,
                                                     training,
                                                     look_ahead_mask,
                                                     padding_mask)
      
    attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
    attention_weights['decoder_layer{}_block2'.format(i+1)] = block2
    
    p_logits, p_probs, p_index, p_value = self.last_decoder_layer(dec_input,
                                                                  enc_input,
                                                                  enc_output,
                                                                  training,
                                                                  attention_mask,
                                                                  look_ahead_mask,
                                                                  padding_mask
                                                                  )


    return p_logits, p_probs, p_index, p_value
