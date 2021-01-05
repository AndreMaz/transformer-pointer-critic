import tensorflow as tf
from agents.models.transformer.actor.decoder import Decoder
from agents.models.transformer.common.encoder import Encoder

class ActorTransformer(tf.keras.Model):
  def __init__(self,
               num_layers,
               d_model,
               num_heads,
               dff,
               positional_encoding: bool,
               vocab_size,
               SOS_CODE,
               encoder_embedding_time_distributed,
               attention_dense_units,
               rate=0.1
               ):

    super(ActorTransformer, self).__init__()

    self.encoder = Encoder(num_layers,
                           d_model,
                           num_heads,
                           dff,
                           positional_encoding,
                           vocab_size,
                           encoder_embedding_time_distributed,
                           rate)

    self.decoder = Decoder(num_layers, 
                           d_model,
                           num_heads,
                           dff,
                           positional_encoding,
                           SOS_CODE,
                           vocab_size,
                           encoder_embedding_time_distributed,
                           attention_dense_units,
                           rate)

    # self.final_layer = tf.keras.layers.Dense(target_vocab_size)
  
  @tf.function
  def call(self,
           enc_input,
           dec_input,
           attention_mask,
           training: bool,
           enc_padding_mask = None,
           look_ahead_mask = None,
           dec_padding_mask = None,
           ):
    
    # enc_output.shape = (batch_size, inp_seq_len, d_model)
    enc_output = self.encoder(enc_input,
                              training,
                              enc_padding_mask)
    
    # Compute a single pointer
    # p_logits.shape = (batch_size, inp_seq_len)
    # p_probs.shape = (batch_size, inp_seq_len)
    # p_index = int
    # p_value.shape = (batch_size, 2)
    p_logits, p_probs, p_index, p_value = self.decoder(dec_input,
                                                       enc_input,
                                                       enc_output,
                                                       attention_mask,
                                                       training,
                                                       look_ahead_mask,
                                                       dec_padding_mask)

    return p_logits, p_probs, p_index, p_value
