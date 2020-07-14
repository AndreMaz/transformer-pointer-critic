from tensorflow.keras.layers import Embedding, LSTM, TimeDistributed, Dense
from tensorflow.keras import Model


class Encoder(Model):
    def __init__(self,
                 embedding_size: int,
                 embedding_time_distributed: bool,
                 lstm_units: int
                 ):

        super(Encoder, self).__init__()

        self.embedding_size = embedding_size
        self.embedding_time_distributed = embedding_time_distributed
        self.lstm_units = lstm_units

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

        self.lstm = LSTM(self.lstm_units,
                       return_sequences=True,
                       return_state=True)

    def call(self, input_state):
        # Get the embeddings for the input
        x = self.embedding(input_state)

        # Run in trough the LSTM
        out, hidden_state, carry_state = self.lstm(x)
        return out, hidden_state, carry_state
