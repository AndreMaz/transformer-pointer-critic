import tensorflow as tf
from agents.models.transformer.common.encoder import Encoder
from agents.models.transformer.common.utils import get_initializer

class CriticTransformer(tf.keras.Model):
    def __init__(self,
                 num_layers,
                 d_model,
                 num_heads,
                 dff,
                 vocab_size,
                 embedding_time_distributed,
                 last_layer_units,
                 last_layer_activation,
                 use_default_initializer:bool = True
                 ):
        super(CriticTransformer, self).__init__()

        # Store the values
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.vocab_size = vocab_size
        self.embedding_time_distributed = embedding_time_distributed

        self.use_default_initializer = use_default_initializer
        self.initializer = get_initializer(self.d_model, self.use_default_initializer)

        self.encoder = Encoder(self.num_layers,
                               self.d_model,
                               self.num_heads,
                               self.dff,
                               self.vocab_size,
                               self.embedding_time_distributed,
                               use_default_initializer
                               )
        
        self.flat_layer = tf.keras.layers.Flatten()

        self.final_layer0 = tf.keras.layers.Dense(last_layer_units,
                                                  activation=last_layer_activation,
                                                  kernel_initializer=self.initializer
                                                  )
        self.final_layer1 = tf.keras.layers.Dense(last_layer_units,
                                                  activation=last_layer_activation,
                                                  kernel_initializer=self.initializer
                                                  )

        self.final_layer = tf.keras.layers.Dense(1, kernel_initializer=self.initializer)
    
    @tf.function
    def call(self,
             encoder_input,
             training: bool,
             num_bins: int,
             enc_padding_mask
             ):

        # Encode the input state
        # enc_output.shape = (batch_size, inp_seq_len, d_model)
        enc_output = self.encoder(
            encoder_input,
            training,
            num_bins,
            enc_padding_mask = enc_padding_mask
        )

        # flatten_output.shape = (batch_size, inp_seq_len * d_model)
        flatten_output = self.flat_layer(enc_output)

        # Pass trough first dense layer
        final_out = self.final_layer0(flatten_output)
        
        # Pass trough second dense layer
        final_out = self.final_layer1(final_out)

        # Generate an estimated state value
        # state_value.shape = (1, 1)
        state_value = self.final_layer(final_out)

        return state_value
