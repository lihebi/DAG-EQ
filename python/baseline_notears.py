#!/usr/bin/env python3

import h5py
import pandas as pd
import tempfile
import numpy as np
import random

from notears.linear import notears_linear
from notears.nonlinear import NotearsMLP, notears_nonlinear

from notears import utils as ut

def test_notears():
    # read X
    os.chdir('julia/src')
    fname = 'data/SF-10/d=10_k=1_gtype=SF_noise=Gaussian_mat=COR.hdf5'
    f = h5py.File(fname, 'r')
    raw_x = f['raw_x']
    raw_y = f['raw_y']

    x = raw_x[0]
    y = raw_y[0]

    # CAUTION should be channel first
    W_est = notears_linear(x.transpose(), lambda1=0, loss_type='l2')
    assert ut.is_dag(W_est)
    np.savetxt('W_est.csv', W_est, delimiter=',')
    acc = ut.count_accuracy(y.transpose(), W_est != 0)
    print(acc)


def run_notears(npx, npy):
    W_est = notears_linear(npx, lambda1=0, loss_type='l2')
    assert ut.is_dag(W_est)
    # np.savetxt('W_est.csv', W_est, delimiter=',')
    # acc = ut.count_accuracy(y.transpose(), W_est != 0)
    


def test_notears_nonlinear():
    os.chdir('julia/src')
    fname = 'data/SF-10/d=10_k=1_gtype=SF_noise=Gaussian_mat=COR_mec=MLP.hdf5'
    f = h5py.File(fname, 'r')
    raw_x = f['raw_x']
    raw_y = f['raw_y']

    x = raw_x[0].transpose()
    y = raw_y[0].transpose()
    x.astype(np.float32)

    d = 10
    model = NotearsMLP(dims=[d, 10, 1], bias=True)
    # (HEBI: seems to be very slow), the example they tried is (200,5) while
    # here it is (1000,10). And yet it finishes in 20 seconds, with very bad
    # results: {'fdr': 0.6666666666666666, 'tpr': 0.3333333333333333, 'fpr':
    # 6.0, 'shd': 15, 'nnz': 9}
    #
    # Wait, fpr = 6.0??
    W_est = notears_nonlinear(model, x.astype(np.float32), lambda1=0.01, lambda2=0.01)
    assert ut.is_dag(W_est)
    np.savetxt('W_est.csv', W_est, delimiter=',')
    acc = ut.count_accuracy(B_true, W_est != 0)
    print(acc)

def test_notears_random():
    n, d, s0, graph_type, sem_type = 100, 20, 20, 'ER', 'gauss'
    B_true = ut.simulate_dag(d, s0, graph_type)
    W_true = ut.simulate_parameter(B_true)
    np.savetxt('W_true.csv', W_true, delimiter=',')

    X = ut.simulate_linear_sem(W_true, n, sem_type)
    np.savetxt('X.csv', X, delimiter=',')

    W_est = notears_linear(X, lambda1=0.1, loss_type='l2')
    assert ut.is_dag(W_est)
    np.savetxt('W_est.csv', W_est, delimiter=',')
    acc = ut.count_accuracy(B_true, W_est != 0)
    print(acc)

def test_notears_nonlinear_random():
    torch.set_default_dtype(torch.double)
    np.set_printoptions(precision=3)

    n, d, s0, graph_type, sem_type = 200, 5, 9, 'ER', 'mim'
    B_true = ut.simulate_dag(d, s0, graph_type)
    np.savetxt('W_true.csv', B_true, delimiter=',')

    X = ut.simulate_nonlinear_sem(B_true, n, sem_type)
    np.savetxt('X.csv', X, delimiter=',')

    model = NotearsMLP(dims=[d, 10, 1], bias=True)
    W_est = notears_nonlinear(model, X.astype(np.float32), lambda1=0.01, lambda2=0.01)
    assert ut.is_dag(W_est)
    np.savetxt('W_est.csv', W_est, delimiter=',')
    acc = ut.count_accuracy(B_true, W_est != 0)
    print(acc)
