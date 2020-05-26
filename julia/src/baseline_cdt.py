#!/usr/bin/env python3
import h5py
import pandas as pd
import tempfile
import random

import cdt
# cdt.SETTINGS
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from cdt.data import load_dataset
from cdt.causality.pairwise import ANM, NCC, RCC
from cdt.causality.graph import GES, LiNGAM, PC, SAM, CAM

def test_networkx():
    g = nx.DiGraph()  # initialize a directed graph
    l = list(g.nodes())  # list of nodes in the graph
    a = nx.adj_matrix(g).todense()  # Output the adjacency matrix of the graph
    e = list(g.edges())  # list of edges in the graph


def test_ANM():
    data, labels = load_dataset('tuebingen')
    obj = ANM()
    
    # This example uses the predict() method
    # (HEBI: this is too slow)
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

    # (HEBI: I'll need to compare with this)
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
    dfx, dfy = construct_df(raw_x, raw_y)

    obj = RCC()
    # why fitting is even easier than predicting
    obj.fit(dfx, dfy)
    # test for performance
    pred = obj.predict(dfx)
    type(pred)
    pred_v = np.array(pred).reshape(-1)
    y_v = np.array(dfy).reshape(-1).shape
    # wrong for half on training data
    (pred_v == y_v)


def construct_df(raw_x, raw_y):
    # the data format should be:
    #
    # X: cols: variables
    #    rows: name: pairID, value: vector for each variable
    # Y: cols: target, 0 or 1
    #    rows: name: pairID, it should be "whether there's an edge from A to B?"
    #
    # UPDATE I'm going to add to the label three values
    # -1 for a<-b
    # 0 for no relation
    # 1 for a->b
    #
    # UPDATE I'm going to use all edges in the graph to make the training
    # corpus, specifically those with a->b edges. That means, all the targets
    # would be 1.0, to match the turburgen pair data. Maybe the internal of the
    # model would automatically train on the reverse edge? And I'm assuming a
    # threshold to be no-relation.
    #
    # I'll use 10 graphs from raw_x to make training pairs, and use the 10
    # graphs for testing
    #
    ct = 0
    outdf = pd.DataFrame(columns={'A', 'B'})
    outY = pd.DataFrame(columns={'target'})

    num = raw_x.shape[0]
    indices = random.sample(range(num), 20)
    # FIXME the training and testing are from the same set of graphs, so the
    # testing result is actually the training one
    test_indices = random.sample(range(num), 20)
    for i in indices:
        x = raw_x[i]
        y = raw_y[i]
        for a in range(y.shape[0]):
            for b in range(y.shape[1]):
                if y[a,b] == 1:
                    # pair (a,b), label 1
                    name = "pair{}".format(ct)
                    print(name)
                    ct+=1

                    # FIXME how to construct pandas dataframe row by row?
                    # FIXME how the name works?
                    outdf.loc[name] = pd.Series({'A': np.array(df[a]), 'B': np.array(df[b])})
                    # FIXME cannot get append working
                    #
                    # outdf.append(
                    #     [np.array(df[a]), np.array(df[b])],
                    #     # {'A': np.array(df[a]), 'B': np.array(df[b])},
                    #     ignore_index=True)
                    #
                    # construct target
                    outY.loc[name] = 1.0
    return outdf, outY

def test_RCC():
    data, labels = load_dataset('tuebingen')
    X_tr, X_te, y_tr, y_te = train_test_split(data, labels, train_size=.5)

    # why all training data has label 1.0?
    
    obj = RCC()
    obj.fit(X_tr, y_tr)
    # This example uses the predict() method
    output = obj.predict(X_te)

    # (HEBI: and this as well)
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

    # (HEBI: this requires pcalg, kpcalg, and
    # https://github.com/Diviyan-Kalainathan/RCIT)
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
    # (HEBI: this seems to be very slow, 6+ hours)
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
