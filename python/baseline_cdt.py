#!/usr/bin/env python3
import h5py
import pandas as pd
import tempfile
import random
import math
from tqdm import tqdm
import numpy as np
import time

import cdt
# cdt.SETTINGS
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier as CLF

import torch
from torch import nn, optim

from cdt.data import load_dataset
from cdt.causality.pairwise import ANM, NCC, RCC
from cdt.causality.graph import GES, LiNGAM, PC, SAM, CAM

from baseline_common import compute_metrics

def test_networkx():
    g = nx.DiGraph()  # initialize a directed graph
    l = list(g.nodes())  # list of nodes in the graph
    a = nx.adj_matrix(g).todense()  # Output the adjacency matrix of the graph
    e = list(g.edges())  # list of edges in the graph


def test_ANM():
    data, labels = load_dataset('tuebingen')
    obj = ANM()
    
    # This example uses the predict() method
    # NOTE This is too slow
    output = obj.predict(data)
    
    # This example uses the orient_graph() method. The dataset used
    # can be loaded using the cdt.data module
    data, graph = load_dataset('sachs')
    output = obj.orient_graph(data, nx.DiGraph(graph))
    
    # To view the directed graph run the following command
    nx.draw_networkx(output, font_size=8)
    plt.show()

def test_NCC():
    data, labels = load_dataset('tuebingen')
    X_tr, X_te, y_tr, y_te = train_test_split(data, labels, train_size=.5)
    
    obj = NCC()
    obj.fit(X_tr, y_tr)
    # This example uses the predict() method
    output = obj.predict(X_te)

    # NOTE: I'll need to compare with this
    # This example uses the orient_graph() method. The dataset used
    # can be loaded using the cdt.data module
    data, graph = load_dataset("sachs")
    output = obj.orient_graph(data, nx.Graph(graph))
    
    #To view the directed graph run the following command
    nx.draw_networkx(output, font_size=8)
    plt.show()



def test():
    os.chdir('julia/src')
    fname = 'data/SF-10/d=10_k=1_gtype=SF_noise=Gaussian_mat=COR.hdf5'
    f = h5py.File(fname, 'r')
    raw_x = f['raw_x']
    raw_y = f['raw_y']
    x = raw_x[0]
    y = raw_y[0]
    df = pd.DataFrame(x.transpose())
    # 1. make training corpus
    train_dfx, train_dfy, test_dfx, test_dfy = construct_df(raw_x, raw_y)
    # train

    # test for performance
    #
    # CAUTION this is the training corpus
    #
    # FIXME this is even slower than training
    sum(np.array(test_dfy ==1))
    test_dfx[np.array(test_dfy == 1).ravel()]
    pred = obj.predict(test_dfx[np.array(test_dfy == 1).ravel()].sample(20))
    pred = obj.predict(test_dfx[np.array(test_dfy == 0).ravel()].sample(20))
    pred = obj.predict(train_dfx[np.array(train_dfy == 1).ravel()].sample(20))
    pred = obj.predict(train_dfx[np.array(train_dfy == 0).ravel()].sample(20))

    # NOTE: it cannot even fit the training data well
    #
    # TODO I'm going to implement a neural network to fit the feature vector
    obj.predict(aug_dfx[np.array(aug_dfy == 1).ravel()].sample(20))
    obj.predict(aug_dfx[np.array(aug_dfy == 0).ravel()].sample(20))


    obj.predict_NN_preprocess(aug_dfx[np.array(aug_dfy == 1).ravel()].sample(20))
    obj.predict_NN_preprocess(aug_dfx[np.array(aug_dfy == 0).ravel()].sample(20))

    pred = obj.predict(test_dfx[0:10])
    type(pred)
    pred_v = np.array(pred).reshape(-1)
    y_v = np.array(dfy).reshape(-1).shape
    # wrong for half on training data
    (pred_v == y_v)

    tmp = [obj.featurize_row(row.iloc[0],
                             row.iloc[1]) for idx, row in aug_dfx[0:8].iterrows()]

    train = np.array([obj.featurize_row(row.iloc[0],
                                        row.iloc[1])
                                     for idx, row in aug_dfx.iterrows()])

class MyRCC(RCC):
    def __init__(self):
        super().__init__()
    def preprocess(self, dfx, dfy):
        # this is very slow, so I'm adding a separete method for computing this
        print('constructing x (featurizing might be very slow) ..')
        # FIXME this is very slow
        x = np.vstack((np.array([self.featurize_row(row.iloc[0],
                                                    row.iloc[1])
                                     for idx, row in dfx.iterrows()]),))
        print(x.shape)
        print('constructing labels ..')
        y = np.vstack((dfy,)).ravel()
        return x, y
    def fit(self, x, y):
        # CAUTION this x and y should not be dataframe, but preprocessed above
        print('training CLF ..')
        verbose = 1 if self.verbose else 0
        # FIXME and this is very im-balanced
        self.clf = CLF(verbose=verbose,
                       min_samples_leaf=self.L,
                       n_estimators=self.E,
                       max_depth=self.max_depth,
                       n_jobs=self.njobs).fit(x, y)
    def fit_NN(self, x, y, num_epochs=1000):
        d = x.shape[1]

        # tx = torch.Tensor(x)
        # ty = torch.Tensor(y).type(torch.long)
        # ty = torch.Tensor(y)

        model = nn.Sequential(nn.Linear(d, 100),
                              nn.Sigmoid(),
                              nn.Linear(100, 1),
                              nn.Sigmoid())
        self.fc = model

        # fit the fc model
        opt = optim.Adam(model.parameters(), lr=1e-3)
        # FIXME whehter to apply sigmoid first for this loss?
        # FIXME binary or n-class?
        # loss_fn = nn.CrossEntropyLoss()

        # FIXME this requires y to be float
        # FIXME do this need sigmoid?
        loss_fn = nn.BCELoss()

        for i in tqdm(range(num_epochs)):
            outputs = nn.Sigmoid()(model(torch.Tensor(x)))
            loss = loss_fn(outputs, torch.unsqueeze(torch.Tensor(y), 1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            running_loss = loss.item()
            # UPDATE disabled because this is distracting from tqdm progressbar
            # if i % 100 == 0:
            #     print('running loss:', running_loss)

    def predict(self, npx):
        _dfx = pd.DataFrame(npx.transpose())
        _dfx = construct_df_mat(_dfx)
        print('featurizing x ..')
        _x = np.vstack((np.array([self.featurize_row(row.iloc[0],
                                                    row.iloc[1])
                                     for idx, row in _dfx.iterrows()]),))
        # run on this
        mat = self.clf.predict(_x)
        # FIXME change this into a adjacency matrix
        d = int(math.sqrt(mat.shape[0]))
        
        mat = mat.reshape(d, d)
        # set diagonal to 0
        np.diag(mat)
        np.fill_diagonal(mat, 0)
        # TODO return networkx graph instance
        graph = nx.DiGraph(mat)
        
        # (outputs.squeeze() > 0.5).numpy().astype(np.int)
        return graph

    def predict_NN(self, npx):
        # I want a whole graph to be predicted
        # npx = test_x[0]
        _dfx = pd.DataFrame(npx.transpose())
        _dfx = construct_df_mat(_dfx)
        print('featurizing x ..')
        _x = np.vstack((np.array([self.featurize_row(row.iloc[0],
                                                     row.iloc[1])
                                     for idx, row in _dfx.iterrows()]),))
        # I'll directly return the adjacency matrix
        # FIXME or just return a networkx graph instance?
        tx = torch.Tensor(_x)
        outputs = self.fc(tx).detach().numpy()
        mat = outputs.squeeze() > 0.5
        # FIXME change this into a adjacency matrix
        d = int(math.sqrt(mat.shape[0]))
        
        mat = mat.reshape(d, d)
        # set diagonal to 0
        np.diag(mat)
        np.fill_diagonal(mat, 0)
        # TODO return networkx graph instance
        graph = nx.DiGraph(mat)
        
        # (outputs.squeeze() > 0.5).numpy().astype(np.int)
        return graph
    
def balance_df(dfx, dfy):
    dfx.shape
    dfy.shape
    # 10% is 1, I'm going to make it 50% by duplicate 1 by 5
    sum(np.array(dfy == 1).ravel())
    one_index = np.array(dfy == 1)
    zero_index = np.array(dfy == 0)

    # OPTION 1: but the trained model seems to be still balanced towards 0, even
    # on training data. Probably because the duplication of data
    #
    # and the data is smaller to train
    aug_dfx = dfx.append([dfx[one_index]] * 9)
    aug_dfy = dfy.append([dfy[one_index]] * 9)

    # OPTION 2: reduce the number of 0 labels
    num_1 = len(dfx[one_index])
    num_0 = len(dfx[zero_index])
    num_1
    num_0

    sample_index = random.sample(range(num_0), num_1)
    aug_dfx = dfx[zero_index].take(sample_index).append(dfx[one_index])
    aug_dfy = dfy[zero_index].take(sample_index).append(dfy[one_index])
    aug_dfx.shape
    aug_dfy.shape
    sum(np.array(aug_dfy == 1))
    return aug_dfx, aug_dfy

def construct_df_mat(x):
    dfx = pd.DataFrame(columns={'A', 'B'})
    d = x.shape[1]
    ct = 0
    for a in range(d):
        for b in range(d):
            name = "pair{}".format(ct)
            ct+=1
            dfx.loc[name] = pd.Series({'A': np.array(x[a]),
                                       'B': np.array(x[b])})
    return dfx

    
def construct_df(raw_x, raw_y):
    # the data format should be:
    #
    # X: cols: variables
    #    rows: name: pairID, value: vector for each variable
    # Y: cols: target, 0 or 1
    #    rows: name: pairID, it should be "whether there's an edge from A to B?"
    #
    # UPDATE the internal automatically train on reverse edge, using -y. I can
    # add 0 as label, and the reverse will be 0 as well, and that duplicated
    # training should be fine.
    #
    # I'll use 10 graphs from raw_x to make training pairs, and use the 10
    # graphs for testing
    #
    ct = 0
    dfx = pd.DataFrame(columns={'A', 'B'})
    dfy = pd.DataFrame(columns={'target'})

    num = raw_x.shape[0]
    indices = random.sample(range(num), 20)
    for i in indices:
        x = raw_x[i]
        y = raw_y[i]

        # FIXME this was not here previously
        df = pd.DataFrame(x.transpose())

        # assuming y.shape[0] == y.shape[1]
        d = y.shape[0]
        for a in range(d):
            for b in range(a):
                # pair (a,b), label 1
                name = "pair{}".format(ct)
                # print(name)
                ct+=1

                # FIXME how to construct pandas dataframe row by row?
                # FIXME how the name works?
                dfx.loc[name] = pd.Series({'A': np.array(df[a]),
                                           'B': np.array(df[b])})
                # FIXME cannot get append working
                #
                # outdf.append(
                #     [np.array(df[a]), np.array(df[b])],
                #     # {'A': np.array(df[a]), 'B': np.array(df[b])},
                #     ignore_index=True)
                #
                # construct target
                # CAUTION this is 0 or 1
                dfy.loc[name] = y[a,b]
    return dfx, dfy

def test_RCC():
    data, labels = load_dataset('tuebingen')
    X_tr, X_te, y_tr, y_te = train_test_split(data, labels, train_size=.5)

    # why all training data has label 1.0?
    
    obj = RCC()
    obj.fit(X_tr, y_tr)
    # This example uses the predict() method
    output = obj.predict(X_te)

    # NOTE: and this as well
    # This example uses the orient_graph() method. The dataset used
    # can be loaded using the cdt.data module
    data, graph = load_dataset('sachs')
    # Oh, this is only used to orient the graph? The graph is already given!
    output = obj.orient_graph(data, nx.DiGraph(graph))
    
    # To view the directed graph run the following command
    nx.draw_networkx(output, font_size=8)
    nx.draw_networkx(graph)
    plt.show()


def test_GES():
    data, graph = load_dataset("sachs")
    obj = GES()
    #The predict() method works without a graph, or with a
    #directed or udirected graph provided as an input
    output = obj.predict(data)    #No graph provided as an argument
    
    output = obj.predict(data, nx.Graph(graph))  #With an undirected graph
    
    output = obj.predict(data, graph)  #With a directed graph
    
    #To view the graph created, run the below commands:
    nx.draw_networkx(output, font_size=8)
    plt.show()

def test_LiNGAM():
    data, graph = load_dataset("sachs")
    obj = LiNGAM()
    output = obj.predict(data)

def test_PC():
    data, graph = load_dataset("sachs")
    # nx.draw_networkx(graph, font_size=8)

    # NOTE: this requires pcalg, kpcalg, and
    # https://github.com/Diviyan-Kalainathan/RCIT
    obj = PC()
    #The predict() method works without a graph, or with a
    #directed or undirected graph provided as an input
    output = obj.predict(data)    #No graph provided as an argument
    
    output = obj.predict(data, nx.Graph(graph))  #With an undirected graph
    
    output = obj.predict(data, graph)  #With a directed graph
    
    #To view the graph created, run the below commands:
    nx.draw_networkx(output, font_size=8)
    plt.show()



def test_SAM():
    data, graph = load_dataset("sachs")
    obj = SAM()
    #The predict() method works without a graph, or with a
    #directed or undirected graph provided as an input
    #
    # NOTE: this seems to be very slow, 6+ hours
    output = obj.predict(data)    #No graph provided as an argument
    
    output = obj.predict(data, nx.Graph(graph))  #With an undirected graph
    
    output = obj.predict(data, graph)  #With a directed graph
    
    #To view the graph created, run the below commands:
    nx.draw_networkx(output, font_size=8)
    plt.show()

def test_CAM():
    data, graph = load_dataset("sachs")
    obj = CAM()
    output = obj.predict(data)
    nx.draw_networkx(output, font_size=8)
    plt.show()


def run_CDT(alg, x, y, vis=True, verbose=False):
    if alg == 'PC':
        obj = PC(alpha=0.1)
    elif alg == 'SAM':
        obj = SAM()
    elif alg == 'CAM':
        # FIXME linear creashes
        # obj = CAM(score='linear')
        obj = CAM(score='nonlinear')
    elif alg == 'GES':
        obj = GES()
    else:
        assert(False)
    obj.verbose = verbose
    output = obj.predict(x)

    # and well, I want to return: prec, recall, shd
    # or, I can just have the adjacency matrix, and compute the matrix myself
    #
    # FIXME the default order seems to be correct
    mat = nx.to_numpy_matrix(output)
    print('adjacency matrix:')
    print(mat)

    if vis:
        nx.draw_networkx(output)
        plt.show()

    return mat

def run_RCC(raw_x, raw_y, model):
    assert model in ['CLF', 'NN']
    # first of all, seperate the data into training and testing
    num = raw_x.shape[0]
    mid = int(round(num * 0.8))
    train_x = raw_x[:mid]
    train_y = raw_y[:mid]
    # FIXME the training and testing are from the same set of graphs, so the
    # testing result is actually the training one
    test_x = raw_x[mid:]
    test_y = raw_y[mid:]

    # create train_df
    dfx, dfy = construct_df(train_x, train_y)
    aug_dfx, aug_dfy = balance_df(dfx, dfy)
    obj = MyRCC()

    # preprocess because it is slow
    x, y = obj.preprocess(aug_dfx, aug_dfy)

    print('fitting ..')
    # FIXME how many epochs
    if model == 'CLF':
        obj.fit(x, y)
    else:
        # FIXME monitor the loss to see whether it is overfitted
        # obj.fit_NN(x, y, 1000)
        print("x.shape", x.shape)
        print("y.shape", y.shape)
        obj.fit_NN(x, y, 10000)

    print('testing ..')
    # predicting
    # TODO predict many
    # randomly sample from test
    #
    # FIXME testing only 10 graphs
    # FIXME keep in-sync with other methods?
    indices = random.sample(range(len(test_x)), 5)
    # TODO featurizing is slow. I'll be doing featurizing all together
    obj.preprocess
    all_prec = 0
    all_recall = 0
    all_shd = 0
    ct = 0
    # FIXME variation is very large, I might want to report this as the downside
    # of this pairwise approach
    #
    # NOTE: the featurization time of our method should be considered as well
    start = time.time()
    for i in indices:
        if model == 'CLF':
            graph = obj.predict(test_x[i])
        else:
            graph = obj.predict_NN(test_x[i])

        # DEBUG draw the graph
        # nx.draw_networkx(graph, font_size=8)
        # plt.show()

        mat = nx.to_numpy_matrix(graph)
        print('adjacency matrix:')
        print(mat)
        # TODO compute the metrics here probably, because I'm sampling some test_dfy
        prec, recall, shd = compute_metrics(mat, test_y[i])
        print('prec:', prec, 'recall:', recall, 'shd:', shd)
        ct += 1
        all_prec += prec
        all_recall += recall
        all_shd += shd
    end = time.time()
    print('Total stats:')
    print('prec:', all_prec/ct,
          'recall:', all_recall/ct,
          'shd:', all_shd/ct)
    return all_prec/ct, all_recall/ct, all_shd/ct, (end-start)/ct
