import time
import pickle
import os
import datetime
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader
import math
import numpy as np
import h5py
from tqdm import tqdm

import networkx as nx

from modules import *
from utils import *

def clear_tqdm():
    if '_instances' in dir(tqdm):
        tqdm._instances.clear()

clear_tqdm()

# The DAG-GNN model mix CPU tensors in the computation, and passing in CUDA tensor would throw errors.
# And the computation is not expensive on CPU anyway.
use_cuda = False

# graph_size = 10
x_dims = 1
z_dims = 1
batch_size = 100
graph_threshold = 0.3

def get_model(graph_size):
    off_diag = np.ones([graph_size, graph_size]) - np.eye(graph_size)

    rel_rec = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float64)
    rel_send = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float64)
    rel_rec = torch.DoubleTensor(rel_rec)
    rel_send = torch.DoubleTensor(rel_send)

    # add adjacency matrix A
    num_nodes = graph_size
    adj_A = np.zeros((num_nodes, num_nodes))
    

    # FIXME use SEM or MLP? for linear
    model_type = 'mlp'
    if model_type == 'mlp':
        encoder = MLPEncoder(graph_size * x_dims, x_dims, 64,
                             z_dims, adj_A,
                             batch_size = batch_size,
                             do_prob = 0, factor = True).double()
        decoder = MLPDecoder(graph_size * x_dims,
                             z_dims, x_dims, encoder,
                             data_variable_size = graph_size,
                             batch_size = batch_size,
                             n_hid=64,
                             do_prob=0).double()
    elif model_type == 'sem':
        encoder = SEMEncoder(graph_size * x_dims, 64,
                             z_dims, adj_A,
                             batch_size = batch_size,
                             do_prob = 0, factor = True).double()

        decoder = SEMDecoder(graph_size * x_dims,
                             z_dims, 2, encoder,
                             data_variable_size = graph_size,
                             batch_size = batch_size,
                             n_hid=64,
                             do_prob=0).double()
    if torch.cuda.is_available() and use_cuda:
        encoder = encoder.cuda()
        decoder = decoder.cuda()
        rel_rec = rel_rec.cuda()
        rel_send = rel_send.cuda()

    rel_rec = Variable(rel_rec)
    rel_send = Variable(rel_send)
    return encoder, decoder, rel_rec, rel_send

# compute constraint h(A) value
def _h_A(A, m):
    expm_A = matrix_poly(A*A, m)
    h_A = torch.trace(expm_A) - m
    return h_A

def stau(w, tau):
    prox_plus = torch.nn.Threshold(0.,0.)
    w1 = prox_plus(torch.abs(w)-tau)
    return torch.sign(w)*w1


def update_optimizer(optimizer, original_lr, c_A):
    '''related LR to c_A, whenever c_A gets big, reduce LR proportionally'''
    MAX_LR = 1e-2
    MIN_LR = 1e-4

    estimated_lr = original_lr / (math.log10(c_A) + 1e-10)
    if estimated_lr > MAX_LR:
        lr = MAX_LR
    elif estimated_lr < MIN_LR:
        lr = MIN_LR
    else:
        lr = estimated_lr

    # set LR
    for parame_group in optimizer.param_groups:
        parame_group['lr'] = lr

    return optimizer, lr


def train(encoder, decoder, rel_rec, rel_send,
          optimizer, scheduler,
          train_loader,
          epoch, best_val_loss, ground_truth_G,
          lambda_A, c_A):
    t = time.time()
    nll_train = []
    kl_train = []
    mse_train = []
    shd_trian = []

    encoder.train()
    decoder.train()
    scheduler.step()


    # update optimizer
    optimizer, lr = update_optimizer(optimizer, 3e-3, c_A)


    # FIXME seems the dataset loader always return a list
    for _, (data,) in enumerate(train_loader):
        if torch.cuda.is_available() and use_cuda:
            data = data.cuda()
        graph_size = data.shape[1]
        data = Variable(data).double()

        optimizer.zero_grad()

        # logits is of size: [num_sims, z_dims]
        (enc_x, logits, origin_A, adj_A_tilt_encoder,
         z_gap, z_positive, myA, Wa) = encoder(data, rel_rec, rel_send)

        edges = logits

        dec_x, output, adj_A_tilt_decoder = decoder(data, edges,
                                                    graph_size * x_dims,
                                                    rel_rec, rel_send,
                                                    origin_A, adj_A_tilt_encoder, Wa)

        if torch.sum(output != output):
            print('nan error\n')
            assert(False)

        target = data
        preds = output
        variance = 0.

        # reconstruction accuracy loss
        loss_nll = nll_gaussian(preds, target, variance)

        # KL loss
        loss_kl = kl_gaussian_sem(logits)

        # ELBO loss:
        loss = loss_kl + loss_nll

        # compute h(A)
        h_A = _h_A(origin_A, graph_size)
        loss += (lambda_A * h_A
                 + 0.5 * c_A * h_A * h_A
                 + 100. * torch.trace(origin_A*origin_A))

        loss.backward()
        loss = optimizer.step()

        myA.data = stau(myA.data, 0)

        if torch.sum(origin_A != origin_A):
            print('nan error\n')

        # compute metrics
        graph = origin_A.data.clone().cpu().numpy()
        graph[np.abs(graph) < graph_threshold] = 0

        fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G, nx.DiGraph(graph))

        mse_train.append(F.mse_loss(preds, target).item())
        nll_train.append(loss_nll.item())
        kl_train.append(loss_kl.item())
        # shd_trian.append(shd)
        # print('fdr:', fdr, 'tpr:', tpr, 'shd:', shd)

    # print(h_A.item())
    nll_val = []
    acc_val = []
    kl_val = []
    mse_val = []

    one = np.mean(np.mean(kl_train)  + np.mean(nll_train))
    return one, np.mean(nll_train), np.mean(mse_train), graph, origin_A

def main():
    for gtype in ['SF', 'ER']:
        for d in [10, 20, 50, 100]:
            prefix = '/home/hebi/git/supervised-causal/julia/src/'
            fname = prefix + 'data/{}-{}/d={}_k=1_gtype={}_noise=Gaussian_mat=COR.hdf5'.format(gtype, d, d, gtype)
            print('== processing', fname, '..')
            dag_gnn(fname, d)
    
def dag_gnn(d, npx, npy, max_iter=5, num_epochs=100):
    # fname = '/data/SF-10/d=10_k=1_gtype=SF_noise=Gaussian_mat=COR.hdf5'
    t_total = time.time()
    best_ELBO_loss = np.inf
    best_NLL_loss = np.inf
    best_MSE_loss = np.inf
    best_epoch = 0
    best_ELBO_graph = []
    best_NLL_graph = []
    best_MSE_graph = []

    encoder, decoder, rel_rec, rel_send = get_model(d)
    optimizer = optim.Adam(list(encoder.parameters())
                           + list(decoder.parameters()),
                           lr=3e-3)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=200,
                                    gamma=1.0)
    
    # optimizer step on hyparameters
    c_A = 1
    lambda_A = 0.0
    h_A_new = torch.tensor(1.)
    h_tol = 1e-8
    # k_max_iter = 100
    k_max_iter = max_iter
    h_A_old = np.inf
    # num_epochs = 300
    num_epochs = num_epochs

    # train_loader, _, _, ground_truth_G = load_data(args, args.batch_size, args.suffix)

    # f = h5py.File(fname, 'r')
    # raw_x = f['raw_x']
    # raw_y = f['raw_y']

    # TODO random sample 10 graphs
    # FIXME how does this model work? Do I need to train once or multiple times?
    # X = npx.transpose()
    X = npx
    train_loader = DataLoader(TensorDataset(torch.FloatTensor(np.expand_dims(X, axis=2))),
                              batch_size=batch_size)
    # Y = npy.transpose()
    Y = npy
    ground_truth_G = nx.DiGraph(Y)
    
    for step_k in range(k_max_iter):
        print('step', step_k)
        while c_A < 1e+20:
            for epoch in tqdm(range(num_epochs)):
                # print('epoch', epoch)
                (ELBO_loss, NLL_loss, MSE_loss, graph,
                 origin_A) = train(encoder, decoder,
                                   rel_rec, rel_send,
                                   optimizer, scheduler,
                                   train_loader,
                                   epoch, best_ELBO_loss, ground_truth_G,
                                   lambda_A, c_A)
                if ELBO_loss < best_ELBO_loss:
                    best_ELBO_loss = ELBO_loss
                    best_epoch = epoch
                    best_ELBO_graph = graph

                if NLL_loss < best_NLL_loss:
                    best_NLL_loss = NLL_loss
                    best_epoch = epoch
                    best_NLL_graph = graph

                if MSE_loss < best_MSE_loss:
                    best_MSE_loss = MSE_loss
                    best_epoch = epoch
                    best_MSE_graph = graph

            print("Optimization Finished!")
            print("Best Epoch: {:04d}".format(best_epoch))
            for graph in [best_ELBO_graph, best_NLL_graph, best_MSE_graph]:
                fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G,
                                                         nx.DiGraph(graph))
                print('fdr:', fdr, 'tpr:', tpr, 'shd:', shd)
            if ELBO_loss > 2 * best_ELBO_loss:
                break

            # update parameters
            A_new = origin_A.data.clone()
            h_A_new = _h_A(A_new, d)
            if h_A_new.item() > 0.25 * h_A_old:
                c_A*=10
            else:
                break

            # update parameters
            # h_A, adj_A are computed in loss anyway, so no need to store
        h_A_old = h_A_new.item()
        lambda_A += c_A * h_A_new.item()

        if h_A_new.item() <= h_tol:
            break

    print('The best:')
    # FIXME looks like all these graphs are the same
    for graph in [best_ELBO_graph, best_NLL_graph, best_MSE_graph]:
        fdr, tpr, fpr, shd, nnz = count_accuracy(ground_truth_G,
                                                 nx.DiGraph(graph))
        print('fdr:', fdr, 'tpr:', tpr, 'shd:', shd)

    # return best_ELBO_graph, best_NLL_graph, best_MSE_graph
    # UPDATE this is mat
    return best_ELBO_graph
