'''This script demonstrates how to build a variational autoencoder with Keras.
 #Reference

 https://github.com/keras-team/keras/blob/master/examples/variational_autoencoder.py

 - Auto-Encoding Variational Bayes
   https://arxiv.org/abs/1312.6114
'''
from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import metrics


class Vae:

    def __init__(self, input_dim, batch_size=100, latent_dim=2, intermediate_dim=256, epochs=50, epsilon_std=1.0):

        self.original_dim = input_dim
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.intermediate_dim = intermediate_dim
        self.epochs = epochs
        self.epsilon_std = epsilon_std
        self.vae = None
        self.encoder = None
        self.decoder = None
        self.x = None # Input layer
        self.z_mean = None
        self.decoder_h = None
        self.decoder_mean = None

        self.build_model()

    def sampling(self, args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], self.latent_dim), mean=0., stddev=self.epsilon_std)
        return z_mean + K.exp(z_log_var / 2) * epsilon

    def build_model(self):

        self.x = Input(shape=(self.original_dim,))
        h = Dense(self.intermediate_dim, activation='relu')(self.x)
        self.z_mean = Dense(self.latent_dim)(h)
        z_log_var = Dense(self.latent_dim)(h)


        # note that "output_shape" isn't necessary with the TensorFlow backend
        z = Lambda(self.sampling, output_shape=(self.latent_dim,))([self.z_mean, z_log_var])

        # we instantiate these layers separately so as to reuse them later
        self.decoder_h = Dense(self.intermediate_dim, activation='relu')
        self.decoder_mean = Dense(self.original_dim, activation='sigmoid')
        h_decoded = self.decoder_h(z)
        x_decoded_mean = self.decoder_mean(h_decoded)

        # instantiate VAE model
        self.vae = Model(self.x, x_decoded_mean)

        # Compute VAE loss
        xent_loss = self.original_dim * metrics.binary_crossentropy(self.x, x_decoded_mean)
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(self.z_mean) - K.exp(z_log_var), axis=-1)
        vae_loss = K.mean(xent_loss + kl_loss)

        self.vae.add_loss(vae_loss)
        self.vae.compile(optimizer='rmsprop')
        self.vae.summary()

    def fit(self, x_train, x_test):

        # Train the entire model
        self.vae.fit(x_train,
                     shuffle=True,
                     epochs=self.epochs,
                     batch_size=self.batch_size,
                     validation_data=(x_test, None))

    def encode(self, data):

        # build a model to project inputs on the latent space
        self.encoder = Model(self.x, self.z_mean)

        return self.encoder.predict(data, batch_size=self.batch_size)

    def decode(self, data):

        # build a data generator that can sample from the learned distribution
        decoder_input = Input(shape=(self.latent_dim,))
        _h_decoded = self.decoder_h(decoder_input)
        _x_decoded_mean = self.decoder_mean(_h_decoded)
        generator = Model(decoder_input, _x_decoded_mean)

        return generator.predict(data)