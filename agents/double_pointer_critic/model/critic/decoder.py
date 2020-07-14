import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import TimeDistributed, Dense

class Decoder(tf.keras.Model):
    def __init__(self,
                decoder_units: int,
                decoder_activation: str
                ):
        super(Decoder, self).__init__()

        self.decoder_units = decoder_units
        self.decoder_activation = decoder_activation

        self.first_layer = Dense(
                self.decoder_units,
                activation=self.decoder_activation
            )
        
        self.second_layer = Dense(
                1,
                activation=self.decoder_activation
            )

    def call(self, glimpse):
        x = self.first_layer(glimpse)
        state_value = self.second_layer(x)

        return state_value