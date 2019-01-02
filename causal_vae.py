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

# tf.set_random_seed(seed)
# np.random.seed(seed)

def plot_results(models,
                 data,
                 batch_size=128,
                 model_name="vae_mnist"):
    """Plots labels and MNIST digits as function of 2-dim latent vector

    # Arguments:
        models (tuple): encoder and decoder models
        data (tuple): test data and label
        batch_size (int): prediction batch size
        model_name (string): which model is using this function
    """

    encoder, decoder = models
    x_test, y_test = data
    os.makedirs(model_name, exist_ok=True)

    filename = os.path.join(model_name, "digits_over_latent.png")
    # display a 30x30 2D manifold of digits
    n = 30
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    # (-4,4) because N(0,1) has most value in this range.
    grid_x = np.linspace(-4, 4, n)
    grid_y = np.linspace(-4, 4, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range + 1
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap='Greys_r')
    plt.savefig(filename)
    plt.show()

def myinv(m):
    res = tf.linalg.inv(m)
    # res = m
    return res


def compute_z(w, mu, sigma):
    dim = K.int_shape(mu)[1]

    w = K.reshape(w, (-1, dim, dim))
    lower = tf.matrix_band_part(w, -1, 0)
    diag = tf.matrix_band_part(w, 0, 0)
    z_w = lower - diag
    # z_w = w

    # Compute the distribution for z
    mat_left = myinv(K.eye(dim) - z_w)
    z_mu = K.batch_dot(mat_left, mu)
    
    mat_middle = tf.matrix_diag(tf.square(sigma))
    mat_right = tf.transpose(myinv(K.eye(dim) - z_w), perm=(0,2,1))
    z_sigma = K.batch_dot(K.batch_dot(mat_left, mat_middle), mat_right)
    return z_w, z_mu, z_sigma

def compute_z_numpy_single(w, mu, sigma):
    dim = mu.shape[0]
    w = np.reshape(w, (dim, dim))
    # np.tril([[1,2,3],[4,5,6],[7,8,9]], -1)
    z_w = np.tril(w, -1)
    # Compute the distribution for z
    mat_left = np.linalg.inv(np.eye(dim) - z_w)
    z_mu = np.matmul(mat_left, mu)

    # np.diag([1,2,3])
    mat_middle = np.diag(np.square(sigma))
    mat_right = np.transpose(np.linalg.inv(np.eye(dim) - z_w))
    z_sigma = np.matmul(np.matmul(mat_left, mat_middle), mat_right)
    return z_w, z_mu, z_sigma

def compute_z_numpy(w, mu, sigma):
    res_w = []
    res_mu = []
    res_sigma = []
    for i in range(w.shape[0]):
        w_, mu_, sigma_ = compute_z_numpy_single(w[i], mu[i], sigma[i])
        res_w.append(w_)
        res_mu.append(mu_)
        res_sigma.append(sigma_)
    return np.array(res_w), np.array(res_mu), np.array(res_sigma)

def test():
    w
    sigma
    out = compute_z_numpy_single(w[0], mu[0], sigma[0])
    w[:2].shape[0]
    out = compute_z_numpy(w[:2], mu[:2], sigma[:2])
    out = compute_z_numpy(w, mu, sigma)
    out[0].shape
    out[1].shape
    out[2].shape
        

def multi_sampling(args):
    w, mu, sigma = args
    w, mu, sigma = compute_z(w, mu, sigma)
    batch = K.shape(mu)[0]
    dim = K.int_shape(mu)[1]
    
    epsilon = K.random_normal(shape=(batch, dim))
    B = tf.matrix_diag(sigma)
    mat_left = myinv(K.eye(dim) - w)
    A = K.batch_dot(mat_left, B)
    epsilon = K.random_normal(shape=(batch, dim))
    return mu + K.batch_dot(A, epsilon)

def causal_vae_model(input_shape, latent_dim):
    """Latent dim is the dim of z"""
    intermediate_dim = 512
    inputs = Input(shape=input_shape, name='encoder_input')
    x = Dense(intermediate_dim, activation='relu')(inputs)
    # w = Dense(w_dim, name='causal_weights',
    #           activity_regularizer=regularizers.l1(10e-5))(x)
    # w = Dense(w_dim, name='causal_weights',
    #           kernel_regularizer=regularizers.l2(0.01))(x)
    w = Dense(latent_dim * latent_dim, name='causal_weights')(x)
    mu = Dense(latent_dim, name='causal_mu')(x)
    sigma = Dense(latent_dim, name='causal_sigma')(x)
    # sample, reparameterization trick
    z = Lambda(multi_sampling, name='z')([w, mu, sigma])
    encoder = Model(inputs,
                    # the last is z
                    [w, mu, sigma, z],
                    # z,
                    name='encoder')
    encoder.summary()
    # build decoder model
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x = Dense(intermediate_dim, activation='relu')(latent_inputs)
    outputs = Dense(input_shape[0], activation='sigmoid')(x)
    decoder = Model(latent_inputs, outputs, name='decoder')
    decoder.summary()
    outputs = decoder(encoder(inputs)[-1])
    vae = Model(inputs, outputs, name='causal_vae_mlp')

    # use either one of these loss
    # reconstruction_loss = mse(inputs, outputs)
    reconstruction_loss = binary_crossentropy(inputs,
                                              outputs)
    reconstruction_loss *= input_shape[0]

    z_w, z_mu, z_sigma = compute_z(w, mu, sigma)

    # term1 = K.log(tf.linalg.det(z_sigma))
    term1 = K.log(K.sum(tf.square(sigma)))
    
    term2 = tf.trace(z_sigma)
    term3 = K.reshape(K.batch_dot(z_mu, z_mu, axes=1), shape=(-1,))
    kl_loss = - 0.5 * (1 + term1 - term2 - term3)
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    # vae.compile(optimizer='sgd')
    # vae.compile(optimizer='adam')
    vae.compile(optimizer='rmsprop')
    vae.summary()
    return vae, encoder, decoder
    
def causal_vae_mnist():
    # MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    image_size = x_train.shape[1]
    original_dim = image_size * image_size
    x_train = np.reshape(x_train, [-1, original_dim])
    x_test = np.reshape(x_test, [-1, original_dim])
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    input_shape = (original_dim, )
    latent_dim = 2
    
    vae, encoder, decoder = causal_vae_model(input_shape, 2)
    # vae.summary()
    # encoder.summary()
    vae.fit(x_train, epochs=20, batch_size=128, shuffle=True,
            validation_data=(x_test, None))
    data = (x_test, y_test)
    plot_results((encoder, decoder),
                 data,
                 batch_size=128,
                 model_name="causal_vae")
    
if __name__ == '__main__':
    causal_vae_mnist()
