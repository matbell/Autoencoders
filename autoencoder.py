from keras.layers import Input, Dense
from keras.models import Model
from keras import optimizers
from keras import backend as K


class Autoencoder:

    def create_model(self, input_size, encoding, decoding):

        input_data = Input(shape=(input_size,))

        # Encoding layers

        encoded = encoded = Dense(encoding[0][0], activation=encoding[0][1])(input_data)

        for layer in encoding[1:]:
            encoded = Dense(layer[0], activation=layer[1])(encoded)

        # Decoding layers

        decoded = decoded = Dense(decoding[0][0], activation=decoding[0][1])(encoded)

        for layer in decoding[1:]:
            decoded = Dense(layer[0], activation=layer[1])(decoded)

        # Autoencoder model
        self.autoencoder = Model(input_data, decoded)

        # Encoder model
        self.encoder = Model(input_data, encoded)

        optimizer = optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
        self.autoencoder.compile(optimizer=optimizer, loss='mean_squared_error')

    def fit_model(self, input, epochs=100, batch_size=100):
        self.autoencoder.fit(input, input, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=0)

    def __init__(self, input_size, encoding_layers, decoding_layers):

        # Clear Keras session
        K.clear_session()

        self.autoencoder = None
        self.encoder = None

        self.create_model(input_size=input_size, encoding=encoding_layers, decoding=decoding_layers)
