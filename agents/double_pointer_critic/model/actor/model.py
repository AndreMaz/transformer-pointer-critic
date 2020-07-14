from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
import tensorflow as tf

from agents.double_pointer_critic.model.actor.encoder import Encoder
from agents.double_pointer_critic.model.actor.decoder import Decoder


class ActorModel(Model):
    def __init__(self,
                 SOS_CODE: int,
                 num_items: int,
                 update_mask: bool,
                 learning_rate: float,
                 encoder_embedding_size: int,
                 encoder_embedding_time_distributed: bool,
                 encoder_lstm_units: int,
                 attention_dense_units: int
                 ):
        super(ActorModel, self).__init__()
        
        self.SOS_CODE = SOS_CODE
        self.num_items = num_items
        self.update_mask = update_mask

        self.learning_rate = learning_rate
        self.opt = tf.optimizers.Adam(learning_rate=self.learning_rate)

        # Encoder Configs
        self.encoder_embedding_size = encoder_embedding_size
        self.encoder_embedding_time_distributed = encoder_embedding_time_distributed
        self.encoder_lstm_units = encoder_lstm_units
        
        self.attention_dense_units = attention_dense_units

        # 1. Block: Encoder
        self.encoder = Encoder(
            self.encoder_embedding_size,
            self.encoder_embedding_time_distributed,
            self.encoder_lstm_units
        )

        # 2. Block: Decoder that computes pointers logits
        self.decoder = Decoder(
            self.SOS_CODE,
            self.num_items,
            self.update_mask,
            self.encoder_embedding_size,
            self.encoder_embedding_time_distributed,
            self.encoder_lstm_units,
            self.attention_dense_units
        )
        
    def call(self, encoder_input, dec_input, mask):

        # 1. Encode input
        encoder_outputs,\
        encoder_hidden_state,\
        encoder_carry_state = self.encoder(encoder_input)
        
        # 2. Compute pointer logits
        pointer_logits, pointers_probs, point_index, dec_output = self.decoder(
            encoder_input,
            encoder_outputs,
            dec_input,
            [encoder_hidden_state, encoder_carry_state],
            mask
        )
        
        return pointer_logits, pointers_probs, point_index, dec_output
