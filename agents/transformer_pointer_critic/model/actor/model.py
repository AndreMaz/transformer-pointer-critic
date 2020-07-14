import tensorflow as tf
from agents.transformer_pointer_critic.model.actor.decoder import Decoder
from agents.transformer_pointer_critic.model.common.encoder import Encoder

class ActorTransformer(tf.keras.Model):
  def __init__(self,
               num_layers,
               d_model,
               num_heads,
               dff,
               input_vocab_size,
               target_vocab_size,
               SOS_CODE,
               pe_input,
               pe_target,
               encoder_embedding_time_distributed,
               rate=0.1
               ):

    super(ActorTransformer, self).__init__()

    self.encoder = Encoder(num_layers,
                           d_model,
                           num_heads,
                           dff,
                           input_vocab_size,
                           pe_input,
                           encoder_embedding_time_distributed,
                           rate)

    self.decoder = Decoder(num_layers, 
                           d_model,
                           num_heads,
                           dff,
                           SOS_CODE,
                           target_vocab_size,
                           pe_target,
                           encoder_embedding_time_distributed,
                           rate)

    # self.final_layer = tf.keras.layers.Dense(target_vocab_size)

  def call(self,
           enc_input,
           dec_input,
           training: bool,
           attention_mask,
           enc_padding_mask = None,
           look_ahead_mask = None,
           dec_padding_mask = None,
           ):
    
    # enc_output.shape = (batch_size, inp_seq_len, d_model)
    enc_output = self.encoder(enc_input,
                              training,
                              enc_padding_mask)
    
    p_logits, p_probs, p_index, p_value = self.decoder(dec_input,
                                                       enc_input,
                                                       enc_output,
                                                       training,
                                                       attention_mask,
                                                       look_ahead_mask,
                                                       dec_padding_mask)

    return p_logits, p_probs, p_index, p_value
