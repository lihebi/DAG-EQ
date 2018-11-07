from keras.layers import Input, Dense
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.layers import Flatten, Lambda
from keras.layers import Reshape, Conv2DTranspose
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os


def gen_causal_data():
    """Generate data with existing causal model.
    """
    pass

# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# z = z_mean + sqrt(var)*eps
def vae_sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.

    # Arguments:
        args (tensor): mean and log of variance of Q(z|X)

    # Returns:
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

def sample_z(dim, w, mu, sigma):
    """FIXME
    """
    z = K.zeros(dim)
    for i in range(dim):
        a,b = weight_segment(i)
        W.append(w[a:b] + [0]*(dim-i))
    for i in range(dim):
        batch = K.shape(mu)[0]
        dim = K.int_shape(mu)[1]
        # by default, random_normal has mean=0 and std=1.0
        epsilon = K.random_normal(shape=(batch, dim))
        z.append(np.prod(z, W[i]) + mu + sigma * epsilon)
    return z
        
def weight_segment(n):
    """
    >>> weight_segment(0)
    (0,0)
    >>> weight_segment(1)
    (0,1)
    >>> weight_segment(2)
    (1,3)
    >>> weight_segment(3)
    (3,6)
    >>> weight_segment(4)
    (6,10)
    """
    if n == 0:
        return (0,0)
    else:
        a,b = weight_segment(n-1)
        return b, b+n
    
def causal_vae():
    # MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    image_size = x_train.shape[1]
    original_dim = image_size * image_size
    x_train = np.reshape(x_train, [-1, original_dim])
    x_test = np.reshape(x_test, [-1, original_dim])
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # network parameters
    input_shape = (original_dim, )
    intermediate_dim = 512
    batch_size = 128
    latent_dim = 2
    epochs = 50

    # number of variables, also number of mu and sigma
    causal_latent_dim = 10
    # number of weights
    causal_w_dim = weight_segment(causal_latent_dim-1)[1]
    # causal_var_dim = causal_latent_dim
    # causal_mu_dim = causal_latent_dim
    # causal_sigma_dim  = causal_latent_dim

    # VAE model = encoder + decoder
    # build encoder model
    # 784
    inputs = Input(shape=input_shape, name='encoder_input')
    # 512
    x = Dense(intermediate_dim, activation='relu')(inputs)
    causal_w = Dense(causal_w_dim, name='causal_weights',
                     # FIXME regularizer
                     activity_regularizer=regularizers.l1(10e-5))(x)
    causal_mu = Dense(causal_latent_dim, name='causal_mu')(x)
    causal_sigma = Dense(causal_latent_dim,
                         name='causal_sigma')(x)
    # FIXME sample, reparameterization trick
    z = sample_z(causal_latent_dim, causal_w, causal_mu, causal_sigma)

    # instantiate encoder model
    encoder = Model(inputs,
                    # the last is z
                    [causal_w, causal_mu, causal_sigma, z],
                    name='encoder')
    encoder.summary()
    plot_model(encoder,
               to_file='causal_vae_mlp_encoder.png',
               show_shapes=True)

    # build decoder model
    latent_inputs = Input(shape=(causal_latent_dim,), name='z_sampling')
    x = Dense(intermediate_dim, activation='relu')(latent_inputs)
    outputs = Dense(original_dim, activation='sigmoid')(x)

    # instantiate decoder model
    decoder = Model(latent_inputs, outputs, name='decoder')
    decoder.summary()
    plot_model(decoder, to_file='causal_vae_mlp_decoder.png', show_shapes=True)

    # instantiate VAE model
    # only use the last (z) of the output of encoder
    outputs = decoder(encoder(inputs)[-1])
    vae = Model(inputs, outputs, name='causal_vae_mlp')

    # model and data
    models = (encoder, decoder)
    data = (x_test, y_test)
    
    # use either one of these loss
    # reconstruction_loss = mse(inputs, outputs)
    reconstruction_loss = binary_crossentropy(inputs,
                                              outputs)

    reconstruction_loss *= original_dim
    
    # FIXME kl loss
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')
    vae.summary()
    plot_model(vae,
               to_file='vae_mlp.png',
               show_shapes=True)
    vae.fit(x_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_test, None))
    plot_results(models,
                 data,
                 batch_size=batch_size,
                 model_name="vae_mlp")
    
