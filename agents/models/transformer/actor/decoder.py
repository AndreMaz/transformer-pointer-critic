import tensorflow as tf
from tensorflow.keras.layers import TimeDistributed, Dense
import numpy as np

from agents.models.transformer.actor.decoder_layer import DecoderLayer
from agents.models.transformer.common.utils import positional_encoding, get_initializer

from agents.models.transformer.actor.last_decoder_layer import LastDecoderLayer
from agents.models.transformer.actor.pointer_attention import PointerAttention

class Decoder(tf.keras.layers.Layer):
  def __init__(self,
               num_layers,
               d_model,
               num_heads,
               dff,
               use_positional_encoding,
               SOS_CODE,
               vocab_size,
               logit_clipping_C: int,
               embedding_time_distributed: bool,
               attention_dense_units,
               rate=0.1,
               use_default_initializer: bool = True):
    super(Decoder, self).__init__()

    self.d_model: int = d_model
    self.num_layers: int = num_layers
    self.embedding_time_distributed: bool = embedding_time_distributed
    self.vocab_size: int = vocab_size
    self.use_positional_encoding: bool = use_positional_encoding

    self.SOS_CODE = SOS_CODE

    self.d_model: int = d_model
    self.num_layers: int = num_layers
    
    self.use_default_initializer = use_default_initializer
    self.initializer = get_initializer(self.d_model, self.use_default_initializer)

    # self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
    if self.embedding_time_distributed:
      self.embedding = TimeDistributed(
          Dense(
              self.d_model,
              kernel_initializer=self.initializer
          )
      )
    else:
      self.embedding = Dense(
          self.d_model,
          kernel_initializer=self.initializer
      )

    if self.use_positional_encoding:
      self.pos_encoding = positional_encoding(self.vocab_size, d_model)
    
    self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate, use_default_initializer) 
                       for _ in range(num_layers)]

    # self.last_decoder_layer = LastDecoderLayer(d_model,
    #                                            num_heads,
    #                                            dff,
    #                                            logit_clipping_C,
    #                                            attention_dense_units,
    #                                            rate,
    #                                            use_default_initializer)

    self.last_decoder_layer = PointerAttention(
      attention_dense_units,
      logit_clipping_C,
      use_default_initializer
    )

    self.dropout = tf.keras.layers.Dropout(rate)
    
  def call(self,
           dec_input,
           enc_input,
           enc_output,
           attention_mask,
           training,
           look_ahead_mask,
           padding_mask):

    seq_len = tf.shape(dec_input)[1]
    attention_weights = {}
    
    dec_input = self.embedding(dec_input)  # (batch_size, target_seq_len, d_model)

    if self.use_positional_encoding:
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
                                                                  attention_mask,
                                                                  )

    # p_logits, p_probs, p_index, p_value = self.last_decoder_layer(dec_input,
    #                                                               enc_input,
    #                                                               enc_output,
    #                                                               training,
    #                                                               attention_mask,
    #                                                               look_ahead_mask,
    #                                                               padding_mask
    #                                                               )

    return p_logits, p_probs, p_index, p_value
