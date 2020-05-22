#!/usr/bin/env python3

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

def test_RCC():
    data, labels = load_dataset('tuebingen')
    X_tr, X_te, y_tr, y_te = train_test_split(data, labels, train_size=.5)
    
    obj = RCC()
    obj.fit(X_tr, y_tr)
    # This example uses the predict() method
    output = obj.predict(X_te)

    # (HEBI: and this as well)
    # This example uses the orient_graph() method. The dataset used
    # can be loaded using the cdt.data module
    data, graph = load_dataset('sachs')
    output = obj.orient_graph(data, nx.DiGraph(graph))
    
    # To view the directed graph run the following command
    nx.draw_networkx(output, font_size=8)
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
