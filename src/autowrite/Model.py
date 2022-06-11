import tensorflow as tf
from tensorflow.keras.models import Model
from keras.layers import (
    Bidirectional,
    LSTM,
    Dense,
    Input,
    Masking,
    TimeDistributed
)

class Autoencoder(Model):
    def __init__(self):
        super(Autoencoder, self).__init__()

        self.latent_dim = 83
        self.n_blstm_layers = 5
        self.n_cells = 64
        self.n_features = 5

        self.encoder = self.build_encoder_model()
        self.decoder = self.build_decoder_model()

#        self.load_weights("./weights/autoencoder")

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def build_encoder_model(self):
        model = tf.keras.models.Sequential()

        model.add(Masking(
            input_shape=(None, self.n_features),
            mask_value=tf.constant([0, 0, 0, 0, 0], tf.float32))
        )

        for i in range(self.n_blstm_layers):
            model.add(Bidirectional(LSTM(self.n_cells,
                                         input_shape=(None, self.n_features),
                                         return_sequences = True,
                                         dropout = 0.5),
                                    merge_mode = 'sum'))

        model.add(TimeDistributed(Dense(self.latent_dim, activation = 'softmax')))
        return model

    def build_decoder_model(self):
        model = tf.keras.models.Sequential()

        for i in range(self.n_blstm_layers):
            model.add(Bidirectional(LSTM(self.n_cells,
                                         input_shape=(None, self.n_features),
                                         return_sequences = True,
                                         dropout = 0.5),
                                    merge_mode = 'sum'))

        model.add(TimeDistributed(Dense(self.n_features, activation = 'sigmoid')))
        return model
