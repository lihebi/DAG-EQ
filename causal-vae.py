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

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

def multi_sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.

    # Arguments:
        args (tensor): mean and log of variance of Q(z|X)

    # Returns:
        z (tensor): sampled latent vector
    """
    mu, A = args
    # mu, A = z_mu, A
    dim = K.int_shape(A)[1]
    epsilon = K.random_normal(shape=(-1, dim, 1))
    # epsilon
    # z_mu + tf.matmul(A, epsilon)
    res = mu + tf.matmul(A, epsilon)
    return tf.reshape(res, shape=(-1, dim))

def compute_z(w, mu, sigma):
    dim = K.int_shape(mu)[1]
    # FIXME this fill in clockwise spiral. But this should not matter
    # as long as I'm using it this way consistently
    w_mat = tf.contrib.distributions.fill_triangular(w)
    # w_mat = add_column_right(add_row_above(w_mat))
    # for i in range(latent_dim):
    #     for j in range(i):
    #         K.update(w_mat[i][j], w[int(i * (i+1) / 2 + j)])
    
    # Compute the distribution for z
    # tf.linalg.inv(K.eye(latent_dim) - w_mat)
    z_mu = tf.linalg.matmul(tf.linalg.inv(K.eye(dim) - w_mat),
                            tf.reshape(mu, (-1,dim,1)))
    z_mu = tf.reshape(z_mu, shape=(-1, dim))
    tf.linalg.inv(K.eye(dim) - w_mat)
    tf.matrix_diag(sigma)
    mat_left = tf.linalg.inv(K.eye(dim) - w_mat)
    mat_middle = tf.matrix_diag(tf.square(sigma))
    mat_right = tf.reshape(tf.transpose(tf.linalg.inv(K.eye(dim) - w_mat)),
                           [-1,2,2])
    # this is a covariate matrix, so actually Omega
    z_sigma = tf.matmul(tf.matmul(mat_left, mat_middle), mat_right)
    z_sigma
    return w_mat, z_mu, z_sigma
    

def multi_sampling_v2(args):
    w, mu, sigma = args
    w_mat, z_mu, z_sigma = compute_z(w, mu, sigma)
    # omega = AA^T
    # FIMXE sigma must be positive definite
    A = tf.linalg.cholesky(z_sigma)
    dim = K.int_shape(A)[1]
    epsilon = K.random_normal(shape=(-1, dim, 1))
    # epsilon
    # z_mu + tf.matmul(A, epsilon)
    # FIXME z_mu shape
    res = z_mu + tf.matmul(A, epsilon)
    return tf.reshape(res, shape=(-1, dim))
    

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
    w = Dense(w_dim, name='causal_weights',
                     # FIXME regularizer
                     activity_regularizer=regularizers.l1(10e-5))(x)
    mu = Dense(latent_dim, name='causal_mu')(x)
    sigma = Dense(latent_dim, name='causal_sigma')(x)

    # sample, reparameterization trick
    z = Lambda(multi_sampling_v2,
               output_shape=(latent_dim,),
               name='z')([w, mu, sigma])
    
    # instantiate encoder model
    encoder = Model(inputs,
                    # the last is z
                    # [w, mu, sigma, z],
                    [w, mu, sigma, z],
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

    w_mat, z_mu, z_sigma = compute_z(w, mu, sigma)
    term1 = K.log(tf.linalg.det(z_sigma))
    term2 = tf.trace(z_sigma)
    term3 = tf.reduce_sum(K.dot(z_mu, z_mu), axis=1)
    
    kl_loss = -0.5 * (1 + term1 - term2 - term3)
    
    # Original VAE kl loss
    # kl_loss = 1 + z_sigma - K.square(z_mu) - K.exp(z_sigma)
    # kl_loss = K.sum(kl_loss, axis=-1)
    # kl_loss *= -0.5
    
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
    
