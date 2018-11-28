from keras.layers import Input, Dense
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.layers import Flatten, Lambda
from keras.layers import Reshape, Conv2DTranspose
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K
from keras import regularizers

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

import tensorflow as tf


def gen_causal_data():
    """Generate data with existing causal model.
    """
    pass

def myinv(m):
    res = tf.linalg.inv(m)
    # res = m
    return res

def compute_z(w, mu, sigma):
    dim = K.int_shape(mu)[1]
    # FIXME this fill in clockwise spiral. But this should not matter
    # as long as I'm using it this way consistently
    w_mat = tf.contrib.distributions.fill_triangular(w)
    
    # Compute the distribution for z
    z_mu = K.batch_dot(myinv(K.eye(dim) - w_mat), mu)
    mat_left = myinv(K.eye(dim) - w_mat)
    mat_middle = tf.matrix_diag(tf.square(sigma))
    mat_right = tf.transpose(myinv(K.eye(dim) - w_mat), perm=(0,2,1))
    z_sigma = K.batch_dot(K.batch_dot(mat_left, mat_middle), mat_right)
    
    return w_mat, z_mu, z_sigma
    

# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# z = z_mean + sqrt(var)*eps
def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.

    # Arguments:
        args (tensor): mean and log of variance of Q(z|X)

    # Returns:
        z (tensor): sampled latent vector
    """

    mu, sigma = args
    batch = K.shape(mu)[0]
    dim = K.int_shape(sigma)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return mu + K.exp(0.5 * sigma) * epsilon

def multi_sampling_v2(args):
    w, mu, sigma = args
    w_mat, z_mu, z_sigma = compute_z(w, mu, sigma)
    batch = K.shape(mu)[0]
    dim = K.int_shape(mu)[1]
    
    # # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    # # epsilon = K.batch_dot(w_mat, epsilon)
    # return mu + K.exp(0.5 * sigma) * epsilon + K.sum(w_mat)

    # omega = AA^T
    # FIMXE sigma must be positive definite
    # A = tf.linalg.cholesky(z_sigma)
    # DEBUG
    # A = z_sigma
    cov = tf.matrix_diag(tf.square(sigma))
    AA = tf.linalg.cholesky(cov)
    mat_left = myinv(K.eye(dim) - w_mat)
    A = K.batch_dot(mat_left, AA)
    
    # dim = K.int_shape(A)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    # epsilon
    # z_mu + tf.matmul(A, epsilon)
    # return (?, 2)
    return z_mu + K.batch_dot(A, epsilon)

# def sample(mu, sigma):
#     res = np.zeros(dim)
#     for i in range(dim):
#         batch = K.shape(mu)[0]
#         dim = K.int_shape(mu)[1]
#         # by default, random_normal has mean=0 and std=1.0
#         epsilon = K.random_normal(shape=(batch, dim))
#         res[i] = (mu_z + K.exp(0.5 * sigma_z) * epsilon)
#     return res
    
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
    # latent_dim = 10
    # number of weights
    w_dim = int(latent_dim * (latent_dim + 1) / 2)

    # VAE model = encoder + decoder
    # build encoder model
    # 784
    inputs = Input(shape=input_shape, name='encoder_input')
    # 512
    x = Dense(intermediate_dim, activation='relu')(inputs)
    # w = Dense(w_dim, name='causal_weights',
    #                  # FIXME regularizer
    #                  # activity_regularizer=regularizers.l1(10e-5)
    # )(x)
    w = Dense(w_dim, name='causal_weights')(x)
    mu = Dense(latent_dim, name='causal_mu')(x)
    sigma = Dense(latent_dim, name='causal_sigma')(x)

    # sample, reparameterization trick
    z = Lambda(multi_sampling_v2,
               # output_shape=(latent_dim,),
               name='z')([w, mu, sigma])
    # debug
    # z = Lambda(sampling, name='z')([mu, sigma])
    
    # instantiate encoder model
    encoder = Model(inputs,
                    # the last is z
                    # [w, mu, sigma, z],
                    # [sigma],
                    # DEBUG
                    # [sigma],
                    z,
                    # sigma,
                    name='encoder')
    
    encoder.summary()
    plot_model(encoder,
               to_file='causal_vae_mlp_encoder.png',
               show_shapes=True)

    # build decoder model
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x = Dense(intermediate_dim, activation='relu')(latent_inputs)
    outputs = Dense(original_dim, activation='sigmoid')(x)

    # instantiate decoder model
    decoder = Model(latent_inputs, outputs, name='decoder')
    decoder.summary()
    plot_model(decoder, to_file='causal_vae_mlp_decoder.png', show_shapes=True)

    # instantiate VAE model
    # only use the last (z) of the output of encoder
    # outputs = decoder(encoder(inputs)[-1])
    # DEBUG
    outputs = decoder(encoder(inputs))
    vae = Model(inputs, outputs, name='causal_vae_mlp')

    # model and data
    models = (encoder, decoder)
    data = (x_test, y_test)
    
    # use either one of these loss
    # reconstruction_loss = mse(inputs, outputs)
    reconstruction_loss = binary_crossentropy(inputs,
                                              outputs)

    reconstruction_loss *= original_dim

    _, z_mu, z_sigma = compute_z(w, mu, sigma)
    term1 = K.log(tf.linalg.det(z_sigma))
    term2 = tf.trace(z_sigma)
    # FIXME term3 has (?,1), while other terms have (?,)
    term3 = K.reshape(K.batch_dot(z_mu, z_mu, axes=1), shape=(-1,))
    # term3 = K.sum(tf.tensordot(z_mu, z_mu, axes=1), axis=1)
    # tf.tensordot(z_mu, z_mu, axes=2)

    # K.dot(z_mu, z_mu)
    # K.eval(K.zeros(5))
    # K.eval(K.dot(K.ones((2,3)), K.ones((3,4))))
    # K.dot(K.ones((1,2)), K.ones((2,1))).shape
    # K.eval(K.dot(K.ones((1,2)), K.ones((2,1))))
    # K.batch_dot(K.ones(2), K.ones(2))
    # K.eval(tf.tensordot(tf.ones(2), tf.ones(2), axes=1))
    # K.eval(tf.matmul(tf.ones((2,3)), tf.ones((3,4))))
    # K.eval(tf.tensordot(tf.ones((2,3)), tf.ones((3,4)), axes=1))
    # K.eval(K.ones(2))

    # kl_loss = -0.5 * (1 + term1 - term2 - term3)
    #
    # TODO NOW Seems to be inconsistent with different starts. Are GPU
    # keeping some states?
    kl_loss = - 0.5 * (1 + term1 - term2 - term3)
    # kl_loss = K.mean(K.mean(w_mat, axis=1), axis=1)
    # kl_loss = 1
    
    # Original VAE kl loss
    # kl_loss = 1 + z_sigma - K.square(z_mu) - K.exp(z_sigma)
    # kl_loss = K.sum(kl_loss, axis=-1)
    # kl_loss *= -0.5
    
    # Trying to use tf.distribution and tf.kl_divergence
    # dqz = tf.distributions.Normal(loc=z_mu, scale=z_sigma)
    # dpz = tf.distributions.Normal(loc=0.0, scale=1.0)

    # dqz = tf.contrib.distributions.MultivariateNormalFullCovariance(loc=z_mu, covariance_matrix=z_sigma)
    # dpz = tf.contrib.distributions.MultivariateNormalFullCovariance(loc=tf.zeros(latent_dim), covariance_matrix=tf.matrix_diag(tf.ones(latent_dim)))
    
    # kl_loss = tf.distributions.kl_divergence(dqz, dpz)

    vae_loss = K.mean(reconstruction_loss + kl_loss)
    # DEBUG
    # vae_loss = K.mean(reconstruction_loss)
    vae.add_loss(vae_loss)
    # vae.compile(optimizer='adam')
    vae.compile(optimizer='rmsprop')
    vae.summary()
    encoder.summary()
    decoder.summary()
    plot_model(vae,
               to_file='vae_mlp.png',
               show_shapes=True)
    vae.fit(x_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_test, None))
    # plot_results(models,
    #              data,
    #              batch_size=batch_size,
    #              model_name="vae_mlp")
    
if __name__ == '__main__':
    causal_vae()
