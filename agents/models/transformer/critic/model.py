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
                 dropout_rate=0.1
                 ):
        super(CriticTransformer, self).__init__()

        # By default init in training mode
        self.training = True

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

        self.final_layer = tf.keras.layers.Dense(1)

    def call(self,
             encoder_input,
             training: bool,
             ):

        # Encode the input state
        # enc_output.shape = (batch_size, inp_seq_len, d_model)
        enc_output = self.encoder(encoder_input, training)

        # flatten_output.shape = (batch_size, inp_seq_len * d_model)
        flatten_output = self.flat_layer(enc_output)

        # Generate an estimated state value
        # state_value.shape = (1, 1)
        state_value = self.final_layer(flatten_output)

        return state_value
