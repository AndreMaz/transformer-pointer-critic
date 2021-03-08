import tensorflow as tf
from tensorflow.keras.layers import TimeDistributed, Dense

from agents.models.transformer.common.encoder_layer import EncoderLayer
from agents.models.transformer.common.utils import get_initializer

class Encoder(tf.keras.layers.Layer):
  def __init__(self,
               num_layers,
               d_model,
               num_heads,
               dff,
               vocab_size,
               embedding_time_distributed: bool,
               dropout_rate=0.1,
               use_default_initializer:bool = True):
    super(Encoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers
    self.embedding_time_distributed = embedding_time_distributed
    self.vocab_size = vocab_size

    self.use_default_initializer = use_default_initializer
    self.initializer = get_initializer(self.d_model, self.use_default_initializer)

    # self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
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
    
    self.enc_layers = [EncoderLayer(d_model, num_heads, dff, dropout_rate, use_default_initializer) 
                       for _ in range(num_layers)]
  
    self.enc_layers_bins = [EncoderLayer(d_model, num_heads, dff, dropout_rate, use_default_initializer) 
                       for _ in range(num_layers)]

    self.enc_layers_resources = [EncoderLayer(d_model, num_heads, dff, dropout_rate, use_default_initializer) 
                       for _ in range(num_layers)]

    self.concatenate_layer = tf.keras.layers.Concatenate(axis=1)

    self.dropout = tf.keras.layers.Dropout(dropout_rate)
        
  def call(self, x, training, num_bins, enc_padding_mask = None):

    seq_len = tf.shape(x)[1]
    
    # adding embedding and position encoding.
    x = self.embedding(x)  # (batch_size, input_seq_len, d_model)

    x = self.dropout(x, training=training)

    # Self-Attention over the bins
    encoded_bins = x[:, :num_bins]
    for i in range(self.num_layers):
      encoded_bins = self.enc_layers_bins[i](
        encoded_bins, training, enc_padding_mask[:, :, :, :num_bins]
      )

    # Self-Attention over the resources
    encoded_resources = x[:, num_bins:]
    for i in range(self.num_layers):
      encoded_resources = self.enc_layers_resources[i](
        encoded_resources, training, enc_padding_mask[:, :, :, num_bins:]
      )

    # Concatenate self-attentions of bins and resources
    concatenated_result = self.concatenate_layer(
      [encoded_bins, encoded_resources]
    )

    # Self-Attention over bins and resources
    for i in range(self.num_layers):
      concatenated_result = self.enc_layers[i](concatenated_result, training, enc_padding_mask)
    
    return concatenated_result  # (batch_size, input_seq_len, d_model)
