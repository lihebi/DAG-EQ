#!/usr/bin/env python3

import random
from keras.layers import Input, Dense
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.layers import Conv1D, MaxPooling1D, UpSampling1D
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

from causal_vae import causal_vae_model, compute_z_numpy

# %matplotlib inline

def make_z2x():
    """Return a function that takes z and return x."""
    inputs = Input(shape=(6,), name='encoder_input')
    UpSampling1D(1)(inputs)
    x = Dense((500), activation='relu',
              kernel_initializer='random_uniform',
              bias_initializer='random_uniform')(inputs)
    outputs = Dense((784), activation='sigmoid',
                    kernel_initializer='random_uniform',
                    bias_initializer='random_uniform')(x)
    z2x_model = Model(inputs, outputs, name='z2x')
    def z2x(z):
        # FIXME performance
        z = tf.convert_to_tensor(z, dtype='float32')
        return K.eval(z2x_model(z))
    return z2x


def make_z2x_upsampling():
    encoded = Input(shape=(6,1), name='encoder_input')
    x = Conv1D(8, 3, activation='relu', padding='same')(encoded)
    x = UpSampling1D(3)(x)
    x = Conv1D(8, 3, activation='relu')(x)
    x = UpSampling1D(3)(x)
    x = Conv1D(16, 3, activation='relu')(x)
    x = UpSampling1D(3)(x)
    x = Conv1D(16, 3, activation='relu')(x)
    x = UpSampling1D(3)(x)
    x = Conv1D(16, 3, activation='relu')(x)
    x = UpSampling1D(3)(x)
    # Using conv only, the results are 48 dim
    decoded = Conv1D(1, 3, activation='sigmoid', padding='same')(x)
    # Thus, I'm using Dense to convert it to 784
    # decoded = Dense(784, activation='sigmoid')(Flatten()(x))
    z2x_model = Model(encoded, Flatten()(decoded), name='z2x')
    def z2x(z):
        z = np.expand_dims(z, 2)
        z = tf.convert_to_tensor(z, dtype='float32')
        val = K.eval(z2x_model(z))
        # return np.squeeze(val)
        return val
    return z2x
    

def multi_sample_plain(C, num):
    """Assume z=Cz+N(0,1), we have

    mu = 0
    Sigma = (I-C)^{-1} (I-C)^{-T}
    """
    z_mu = np.zeros(C.shape[0])
    np.eye(C.shape[0])
    tmp = np.linalg.inv(np.eye(C.shape[0]) - C)
    z_sigma = np.matmul(tmp, np.transpose(tmp))
    data = np.random.multivariate_normal(z_mu, z_sigma, (num,))
    return data

def sample_delta_z_single(dim):
    idx = np.random.randint(0, 6)
    delta_z = np.zeros(6)
    delta_z[idx] = np.random.randn()
    return delta_z

def sample_delta_z(dim, num):
    """FIXME Performance"""
    return np.array([sample_delta_z_single(dim) for _ in range(num)])

def test():
    multi_sample_plain(C, 5)


def data_gen(num):
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
    C = np.array([[0,0,0,0,0,0],
                  [1,0,0,0,0,0],
                  [1,2,0,0,0,0],
                  [0,0,0,0,0,0],
                  [0,3,0,1,0,0],
                  [2,0,0,0,0,0]])
    # 1000 data points for z (1000, 6)
    z = multi_sample_plain(C, num)
    # intervene (1000,6)
    delta_z = sample_delta_z(6, num)
    delta_z
    # compute effect ez
    z_prime = z + delta_z
    # ez = z_prime + np.matmul(C, delta_z)
    ez = z_prime + np.array([np.matmul(C, dz) for dz in delta_z])
    z_prime
    ez
    print('different field:', np.sum(ez != z_prime))

    # A = tf.Variable(tf.random_normal(shape=(1000, 3, 4)))
    # B = tf.Variable(tf.random_normal(shape=(1000, 4, 5)))
    # tf.matmul(A, B)

    # generate x=f(z)
    # z2x = make_z2x()
    z2x = make_z2x_upsampling()
    
    # compute x
    x = z2x(z)
    x.shape
    x
    x_prime = z2x(z_prime)
    ex = z2x(ez)
    x_prime
    ex
    delta_x = x_prime - x
    delta_x
    return ((z, delta_z, z_prime, ez),
            batch_batch_normalize((x, delta_x, x_prime, ex)),
            # (x, delta_x, x_prime, ex),
            C)

def mynp_batch_matmul(a, b):
    return np.array([np.matmul(a[i], b[i]) for i in range(a.shape[0])])


def normalize(v):
    # m = np.mean(v)
    # std = np.std(v)
    # vv = (v - m) / std
    # vv[vv<0] = 0
    # vv[vv>1] = 1
    # return vv

    max_v = np.max(v)
    min_v = np.min(v)
    return (v - min_v) / (max_v - min_v)
    

def batch_normalize(v):
    return np.array([normalize(vi) for vi in v])

def batch_batch_normalize(v):
    return [batch_normalize(vi) for vi in v]

def run_synthetic_data():
    (z, delta_z, z_prime, ez), (x, delta_x, x_prime, ex), C = data_gen(10000)
    x.shape
    # split data into test and validate
    num_validation_samples = int(0.1 * x.shape[0])
    x_train = x[:-num_validation_samples]
    x_val = x[-num_validation_samples:]
    # model
    # input_shape = (784,)
    input_shape = (x.shape[1],)
    vae, encoder, decoder = causal_vae_model(input_shape, 6)
    vae.fit(x_train, epochs=20, batch_size=128,
            validation_data=(x_val, None))

    # inference
    # FIXME there will be two sets of w, mu, sigma
    w, mu, sigma, zz = encoder.predict(x)
    w2, _, _, zz_prime = encoder.predict(x_prime)
    w
    w2
    delta_zz = zz_prime - zz
    delta_zz
    zz

    z_w, z_mu, z_sigma = compute_z_numpy(w, mu, sigma)
    # Method 1: compute the distance of w_mat and C
    z_w.shape
    z_w_mean = np.mean(z_w, axis=0)
    z_w_mean
    C
    metric1 = K.eval(K.binary_crossentropy(tf.convert_to_tensor(z_w_mean),
                                           tf.convert_to_tensor(C, dtype='float32')))
    metric1
    np.sum(metric1)
    # ezz = zz_prime + np.matmul(z_w, delta_zz)
    ezz = zz_prime + mynp_batch_matmul(z_w, delta_zz)
    z_w.shape
    delta_zz.shape
    ezz
    ezz.shape

    exx = decoder.predict(ezz)
    exx.shape
    # Method 2: compute the distance of ex and exx
    metric2 = K.eval(K.binary_crossentropy(tf.convert_to_tensor(ex),
                                           tf.convert_to_tensor(exx)))
    print(np.average(metric1))
    print(np.average(metric2))

    # Not using intervention, purely encode xprime to zprime and decode
    exx_prime = decoder.predict(zz_prime)
    metric3 = K.eval(K.binary_crossentropy(tf.convert_to_tensor(exx_prime),
                                           tf.convert_to_tensor(exx)))
    np.average(metric3)
    return
    
