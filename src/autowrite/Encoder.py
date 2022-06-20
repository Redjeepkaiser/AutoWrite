import numpy as np
from pathlib import Path
from Preprocessor import Preprocessor
import pickle
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

class Encoder:
    def __init__(self, path_to_weights, path_to_alphabet):
        self.latent_dim = 83
        self.n_blstm_layers = 5
        self.n_cells = 64
        self.n_features = 11

        self.path_to_weights = Path(path_to_weights)
        print(self.path_to_weights.parent)
        assert self.path_to_weights.parent.is_dir()

        self.preprocessor = Preprocessor(path_to_alphabet)

        self.encoder = self.build_encoder_model()
        self.encoder.load_weights(self.path_to_weights)

    def call(self, x):
        return self.encoder.call(x, mask=None)

    def build_encoder_model(self):
        model = tf.keras.models.Sequential()

        model.add(Masking(
            input_shape=(None, self.n_features),
            mask_value=np.zeros((self.n_features))
        ))

        for i in range(self.n_blstm_layers):
            model.add(Bidirectional(LSTM(self.n_cells,
                                         input_shape=(None, self.n_features),
                                         return_sequences = True,
                                         dropout = 0.5),
                                    merge_mode = 'sum'))

        model.add(TimeDistributed(Dense(self.latent_dim, activation = 'softmax')))
        return model

    def preprocess(self, strokes):
        return self.preprocessor.strokes_to_bezier(strokes)

    def decode_output(self, output):
        print("shape", output.shape)
        print("target", output.shape[1], output.shape[0], output.shape[2])
        reshaped_output = tf.reshape(output,
            (output.shape[1], output.shape[0], output.shape[2]))
        (decoded, log_probs) = tf.nn.ctc_beam_search_decoder(reshaped_output,
                                                             [reshaped_output.shape[0]],
                                                             beam_width=3)
        return "".join(self.preprocessor.decode_sample(decoded[0].values))
