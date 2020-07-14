import tensorflow as tf
from tensorflow.keras.layers import TimeDistributed, Dense

from agents.transformer_pointer_critic.model.common.encoder_layer import EncoderLayer
from agents.transformer_pointer_critic.model.common.utils import positional_encoding

class Encoder(tf.keras.layers.Layer):
  def __init__(self,
               num_layers,
               d_model,
               num_heads,
               dff,
               input_vocab_size,
               maximum_position_encoding,
               embedding_time_distributed: bool,
               dropout_rate=0.1):
    super(Encoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers
    self.embedding_time_distributed = embedding_time_distributed
    self.vocab_size = input_vocab_size

    # self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
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
        
    # Shape is (1, vocab_size, d_model)
    self.pos_encoding = positional_encoding(maximum_position_encoding, 
                                            self.d_model)
    
    
    self.enc_layers = [EncoderLayer(d_model, num_heads, dff, dropout_rate) 
                       for _ in range(num_layers)]
  
    self.dropout = tf.keras.layers.Dropout(dropout_rate)
        
  def call(self, x, training, enc_padding_mask = None):

    seq_len = tf.shape(x)[1]
    
    # adding embedding and position encoding.
    x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    # Add positional encoding
    x += self.pos_encoding[:, :seq_len, :]

    x = self.dropout(x, training=training)
    
    for i in range(self.num_layers):
      x = self.enc_layers[i](x, training)
    
    return x  # (batch_size, input_seq_len, d_model)
