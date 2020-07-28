import tensorflow as tf
from agents.models.transformer.common.encoder import Encoder

class CriticTransformer(tf.keras.Model):
    def __init__(self,
                 num_layers,
                 d_model,
                 num_heads,
                 dff,
                 positional_encoding,
                 vocab_size,
                 embedding_time_distributed,
                 last_layer_units,
                 last_layer_activation,
                 dropout_rate=0.1
                 ):
        super(CriticTransformer, self).__init__()

        # Store the values
        self.num_layers = num_heads
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.vocab_size = vocab_size
        self.positional_encoding = positional_encoding
        self.embedding_time_distributed = embedding_time_distributed
        self.dropout_rate = dropout_rate

        self.encoder = Encoder(self.num_layers,
                               self.d_model,
                               self.num_heads,
                               self.dff,
                               self.positional_encoding,
                               self.vocab_size,
                               self.embedding_time_distributed,
                               self.dropout_rate
                               )
        
        self.flat_layer = tf.keras.layers.Flatten()

        self.final_layer0 = tf.keras.layers.Dense(last_layer_units,
                                                  activation=last_layer_activation
                                                  )
        self.final_layer = tf.keras.layers.Dense(1)

    def call(self,
             encoder_input,
             training: bool,
             enc_padding_mask
             ):

        # Encode the input state
        # enc_output.shape = (batch_size, inp_seq_len, d_model)
        enc_output = self.encoder(
            encoder_input,
            training,
            enc_padding_mask = enc_padding_mask
        )

        # Pass trough first dense layer
        final_out = self.final_layer0(enc_output)

        # flatten_output.shape = (batch_size, inp_seq_len * d_model)
        flatten_output = self.flat_layer(final_out)

        # Generate an estimated state value
        # state_value.shape = (1, 1)
        state_value = self.final_layer(flatten_output)

        return state_value
