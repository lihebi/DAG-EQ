def compute_z_old(w, mu, sigma):
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
def plot_history(history, filename):
    """
    Plot acc and loss in the same figure.
    """
    # filename = 'history.png'
    # Plot training & validation accuracy values
    # Plot training & validation loss values
    # file2 = 'history2.png'
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    # plt.savefig(file2)
    plt.savefig(filename)
    # plt.show()
    # plt.savefig(filename)
    
def plot_multi_history(histories, legends, filename):
    """
    Plot acc and loss in the same figure.
    """
    # filename = 'history.png'
    # Plot training & validation accuracy values
    # Plot training & validation loss values
    # file2 = 'history2.png'
    histories = [history1, history2, history3]
    legends = ['causal', 'causal_reg', 'vae']
    filename = 'history.png'
    plt.figure()
    for h in histories:
        plt.plot(h.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(legends, loc='upper left')
    # plt.savefig(file2)
    plt.savefig(filename)
    # plt.show()
    # plt.savefig(filename)
