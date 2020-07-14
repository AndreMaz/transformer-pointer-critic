from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
import tensorflow as tf

from agents.double_pointer_critic.model.critic.encoder import Encoder
from agents.double_pointer_critic.model.critic.decoder import Decoder
from agents.double_pointer_critic.model.critic.process import Process

class CriticModel(Model):
    def __init__(self,
                 num_items: int,
                 learning_rate: float,
                 encoder_embedding_size: int,
                 encoder_embedding_time_distributed: bool,
                 encoder_lstm_units: int,
                 processing_lstm_units: int,
                 processing_dense_units: int,
                 decoder_units: int,
                 decoder_activation: str
                 ):
        super(CriticModel, self).__init__()
        
        self.num_items = num_items
        self.learning_rate = learning_rate
        self.opt = tf.optimizers.Adam(learning_rate=self.learning_rate)

        # Encoder Configs
        self.encoder_embedding_size = encoder_embedding_size
        self.encoder_embedding_time_distributed = encoder_embedding_time_distributed
        self.encoder_lstm_units = encoder_lstm_units

        # Processing block configs
        self.processing_lstm_units = processing_lstm_units
        self.processing_dense_units = processing_dense_units

        # Decoder block configs
        self.decoder_units = decoder_units
        self.decoder_activation = decoder_activation

        # 1st Block: Encoder
        self.encoder = Encoder(
            self.encoder_embedding_size,
            self.encoder_embedding_time_distributed,
            self.encoder_lstm_units
        )
        
        # 2nd Block: Processing LSMT Glimpse
        self.glimpse_processing = Process(
            self.num_items,
            self.processing_lstm_units,
            self.processing_dense_units
        )

        # 3rd Block: Value estimator
        self.decoder = Decoder(
            self.decoder_units,
            self.decoder_activation
        )
        
    def call(self, encoder_input):
        # 1. Encode input
        encoder_outputs, encoder_hidden_state, encoder_carry_state = self.encoder(
        encoder_input)
        
        # 2. Compute attention
        glimpses = self.glimpse_processing(encoder_outputs, encoder_hidden_state, encoder_carry_state)

        # 3. Compute the estimate value
        state_values = self.decoder(glimpses)
        
        return state_values