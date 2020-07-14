import tensorflow as tf
import numpy as np

from agents.double_pointer_critic.model.actor.pointer_attention import PointerAttention

from tensorflow.keras.layers import TimeDistributed, Dense, LSTM, LSTMCell, Embedding

class Decoder(tf.keras.Model):
    def __init__(self,
                 SOS_CODE: int,
                 num_items: int,
                 update_mask: bool,
                 embedding_size: int,
                 embedding_time_distributed: bool,
                 lstm_units: int,
                 attention_dense_units: int
                 ):
        super(Decoder, self).__init__()

        self.SOS_CODE = SOS_CODE
        self.num_items = num_items
        self.update_mask = update_mask
        
        self.BIG_NUMBER = 1e6

        self.embedding_size = embedding_size
        self.embedding_time_distributed = embedding_time_distributed
        self.lstm_units = lstm_units
        self.attention_dense_units = attention_dense_units

        if (self.embedding_time_distributed):
            self.embedding = TimeDistributed(
            Dense(
                self.embedding_size
            )
        )
        else:
            self.embedding = Dense(
                self.embedding_size
            )

        # We are going to do the looping manually so instead of LSMT Layer we use LSTM cell
        self.cell = LSTMCell(
             self.lstm_units,
             recurrent_initializer='glorot_uniform'
        )

        # Attention Layers
        self.attention = PointerAttention(
            self.attention_dense_units
        )

    def call(self, enc_input, enc_outputs, dec_input, dec_hidden, mask):
        # enc_input.numpy()

        # Create a tensor with the batch indices
        batch_indices = tf.convert_to_tensor(
            list(range(enc_outputs.shape[0])), dtype='int64')

        # Convert input to embeddings
        decoder_embedding = self.embedding(dec_input)

        # Remove time step dim
        decoder_embedding = tf.squeeze(decoder_embedding, axis=1)

        # Call the cell
        step_output, _ = self.cell(
            decoder_embedding,
            states=dec_hidden # Encoder's hidden and carry states
        )

        # Compute the pointers
        pointers_logits = self.attention(step_output, enc_outputs)

        # Apply the mask
        pointers_logits -= mask * self.BIG_NUMBER

        # Apply softmax
        pointers_probs = tf.nn.softmax(pointers_logits, axis=1)

        # Grab the indice of the values pointed by the pointer
        point_index = pointers_probs.numpy().argmax(1)

        # Grab decoded element
        dec_output = enc_input.numpy()[batch_indices, point_index]

        return pointers_logits,\
            pointers_probs,\
            point_index,\
            dec_output,
