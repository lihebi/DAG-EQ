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
    B = tf.matrix_diag(sigma)
    # AA = tf.linalg.cholesky(cov)
    
    mat_left = myinv(K.eye(dim) - w_mat)
    A = K.batch_dot(mat_left, B)
    
    # dim = K.int_shape(A)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    # epsilon
    # z_mu + tf.matmul(A, epsilon)
    # return (?, 2)
    return z_mu + K.batch_dot(A, epsilon)

SAMPLE_DIM = 6

def multi_sample_plain(C, num):
    """Assume z=Cz+N(0,1)"""
    mu_z = 0
    B = K.eval(myinv(K.eye(SAMPLE_DIM) - C))
    epsilon = K.eval(K.random_normal(shape=(num, dim)))
    z = [np.dot(B, e) for e in epsilon]
    return z
    

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

def causal_vae_model(input_shape, latent_dim):
    """Latent dim is the dim of z"""
    intermediate_dim = 512
    # latent_dim = 2
    w_dim = int(latent_dim * (latent_dim + 1) / 2)
    inputs = Input(shape=input_shape, name='encoder_input')
    x = Dense(intermediate_dim, activation='relu')(inputs)
    # w = Dense(w_dim, name='causal_weights',
    #           activity_regularizer=regularizers.l1(10e-5))(x)
    # w = Dense(w_dim, name='causal_weights',
    #           kernel_regularizer=regularizers.l2(0.01))(x)
    w = Dense(w_dim, name='causal_weights')(x)
    mu = Dense(latent_dim, name='causal_mu')(x)
    sigma = Dense(latent_dim, name='causal_sigma')(x)
    # sample, reparameterization trick
    z = Lambda(multi_sampling_v2,
               # output_shape=(latent_dim,),
               name='z')([w, mu, sigma])
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
    # build decoder model
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    x = Dense(intermediate_dim, activation='relu')(latent_inputs)
    outputs = Dense(original_dim, activation='sigmoid')(x)
    decoder = Model(latent_inputs, outputs, name='decoder')
    decoder.summary()
    outputs = decoder(encoder(inputs))
    vae = Model(inputs, outputs, name='causal_vae_mlp')

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
    kl_loss = - 0.5 * (1 + term1 - term2 - term3)
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
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
    vae, encoder, decoder = causal_vae_model(input_shape, 2)
    vae.summary()
    vae.fit(x_train, epochs=10, batch_size=128,
            validation_data=(x_test, None))
    data = (x_test, y_test)
    plot_results((encoder, decoder),
                 data,
                 batch_size=128,
                 model_name="causal_vae")


def make_z2x():
    """Return a function that takes z and return x."""
    inputs = Input(shape=(6), name='encoder_input')
    x = Dense((500), activation='relu',
              kernel_initializer='random_uniform',
              bias_initializer='random_uniform')(inputs)
    outputs = Dense((1000), activation='relu',
                    kernel_initializer='random_uniform',
                    bias_initializer='random_uniform')(x)
    z2x_model = Model(inputs, outputs, name='z2x')
    def z2x(z):
        return z2x_model.predict(z)
    return z2x


def data_gen():
    """Generate high dimentional data according to a low-dimensional
    causal model.

    Assume model (dim 6):
    - z0 = N
    - z1 = z0 + N
    - z2 = z0 + 2 * z1 + N
    - z3 = N
    - z4 = 3 * z1 + z3 + N
    - z5 = 2 * z0 + N

    C= [
    0 0 0 0 0 0
    1 0 0 0 0 0
    1 2 0 0 0 0
    0 0 0 0 0 0
    0 3 0 1 0 0
    2 0 0 0 0 0
    ]

    - First, generate observational data (z0,z1,z2,z3,z4,z5)
    - Generate intervened data:
      - select one entry above (z)
    
      - random select a variable (FIXME only one variable for now),
        mutate to random value (FIXME according to a distribution?)
        (\delta z)
    
      - directly compute new value (z')
      - update all other variables (ez)
    - [optional] Combine to obtain hybrid data

    (z, \delta z, z', ez)

    - Generate a random 2-layer Neural network NN from (6) to (2000),
      use it as $f$
    - compute x = f(z)

    (x, x', ez) => \delta x = x' - x

    Training:
    - Use x as input to causal VAE

    Evaluation:
    - evaluate C and C'
    - Given x and x', try to generate ex by:
      - decode(effect(encode(x)))

    """
    C = np.array([[0,0,0,0,0],
                  [1,0,0,0,0],
                  [1,2,0,0,0,0],
                  [0,0,0,0,0,0],
                  [0,3,0,1,0,0],
                  [2,0,0,0,0,0]])
    # 1000 data points for z
    z = multi_sample_plain(C, 1000)
    # intervene
    idx = randomm(6)
    delta_z = np.zeros((6))
    delta_z[idx] = np.random_normal()
    # compute effect ez
    z_prime = z + delta_z
    ez = z_prime + np.dot(C, delta_z)

    # generate x=f(z)
    z2x = make_z2x()
    # compute x
    x = [z2x(zi) for zi in z]
    x_prime = [z2x(zi) for zi in z_prime]
    ex = [z2x(zi) for zi in ez]
    delta_x = x_prime - x
    return (z, delta_z, z_prime, ez), (x, delta_x, x_prime, ex), C


def run_synthetic_data():
    (z, delta_z, z_prime, ez), (x, delta_x, x_prime, ex), C = data_gen()
    # split data into test and validate
    num_validation_samples = int(0.1 * features.shape[0])
    x_train = x[:-num_validation_samples]
    x_val = x[-num_validation_samples:]
    # model
    input_shape = (1000, )
    model = causal_vae_model(input_shape, 6)
    model.fit(x_train, epochs=epochs, batch_size=batch_size,
              validation_data=(x_val, None))
    # inference
    encoder, decoder = model
    # FIXME there will be two sets of w, mu, sigma
    w, mu, sigma, zz = encoder.predict(x)
    _, _, _, zz_prime = encoder.predict(x_prime)
    delta_zz = zz_prime - zz
    # FIXME w to w_mat
    w_mat, z_mu, z_sigma = compute_z(w, mu, sigma)
    # TODO Method 1: compute the distance of w_mat and C
    metric1 = K.binary_crossentropy(w_mat, C)
    ezz = zz_prime + K.dot(w_mat, delta_zz)
    exx = decoder.predict(ezz)
    # TODO Method 2: compute the distance of ex and exx
    metric2 = K.binary_crossentropy(ex, exx)
    print(metric1)
    print(metric2)
    return
    
    
    
if __name__ == '__main__':
    causal_vae()
