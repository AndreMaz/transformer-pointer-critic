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
               embedding_time_distributed: bool,
               use_default_initializer: bool,
               common_embedding: bool,
               num_bin_features: int,
               num_resource_features: int):

    super(Encoder, self).__init__()

    self.common_embedding = common_embedding
    self.num_bin_features = num_bin_features
    self.num_resource_features = num_resource_features

    self.d_model = d_model
    self.num_layers = num_layers
    self.embedding_time_distributed = embedding_time_distributed

    self.use_default_initializer = use_default_initializer
    self.initializer = get_initializer(self.d_model, self.use_default_initializer)

    if self.embedding_time_distributed:
      self.embedding1 = TimeDistributed(
          Dense(self.d_model, kernel_initializer=self.initializer)
      )
      # If the env state needs separate embedding
      # For example, if features have different meanings
      # Or if, the number of features is different
      if not self.common_embedding:
       self.embedding2 = TimeDistributed(
          Dense(self.d_model, kernel_initializer=self.initializer)
        ) 
    else:
      self.embedding1 = Dense(self.d_model, kernel_initializer=self.initializer)
      # If the env state needs separate embedding
      # For example, if features have different meanings
      # Or if, the number of features is different
      if not self.common_embedding:
        self.embedding2 = Dense(self.d_model, kernel_initializer=self.initializer)
    
    if self.common_embedding:
      self.embedding_fn = self.common_embedding_fn
    else:
      self.embedding_fn = self.unique_embedding_fn

    self.enc_layers = [EncoderLayer(d_model, num_heads, dff, use_default_initializer) 
                       for _ in range(num_layers)]
  
    self.enc_layers_bins = [EncoderLayer(d_model, num_heads, dff, use_default_initializer) 
                       for _ in range(num_layers)]

    self.enc_layers_resources = [EncoderLayer(d_model, num_heads, dff, use_default_initializer) 
                       for _ in range(num_layers)]

    self.concatenate_layer = tf.keras.layers.Concatenate(axis=1)
        
  def call(self, x, training, num_bins, enc_padding_mask = None):

    seq_len = tf.shape(x)[1]
    
    # adding embedding and position encoding.
    # x = self.embedding1(x)  # (batch_size, input_seq_len, d_model)

    encoded_bins, encoded_resources = self.embedding_fn(x, num_bins)

    # Self-Attention over the bins
    # encoded_bins = x[:, :num_bins]
    for i in range(self.num_layers):
      encoded_bins = self.enc_layers_bins[i](
        encoded_bins, training, enc_padding_mask[:, :, :, :num_bins]
      )

    # Self-Attention over the resources
    # encoded_resources = x[:, num_bins:]
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

  def common_embedding_fn(self, x, num_bins):

    # Pass thought embedding layer and then split
    x = self.embedding1(x)

    encoded_bins = x[:, :num_bins]
    encoded_resources = x[:, num_bins:]

    return encoded_bins, encoded_resources
  
  def unique_embedding_fn(self, x, num_bins):
    
    # First split and then pass through embedding layers
    encoded_bins = x[:, :num_bins, :self.num_bin_features]
    encoded_resources = x[:, num_bins:, :self.num_resource_features]

    encoded_bins = self.embedding1(encoded_bins)

    encoded_resources = self.embedding2(encoded_resources)

    return encoded_bins, encoded_resources