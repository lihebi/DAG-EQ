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

SAMPLE_DIM = 6

def multi_sample_plain(C, num):
    """Assume z=Cz+N(0,1)"""
    mu_z = 0
    B = K.eval(myinv(K.eye(SAMPLE_DIM) - C))
    epsilon = K.eval(K.random_normal(shape=(num, dim)))
    z = [np.dot(B, e) for e in epsilon]
    return z
    


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
    
