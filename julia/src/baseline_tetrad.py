#!/usr/bin/env python3

# baselines

import h5py
import pandas as pd
import tempfile
import numpy as np
import random

# I'll make tetrad a global variable
from pycausal.pycausal import pycausal as pcm
from pycausal import search as pcs

pc = None
tetrad = None

def setup():
    global pc
    global tetrad
    pc = pcm()
    pc.start_vm()
    tetrad = pcs.tetradrunner()
    tetrad.listAlgorithms()
    tetrad.listIndTests()
    tetrad.listScores()

def teardown():
    # FIXME it seems that once stopped, it cannot be restarted
    pc.stop_vm()

def run_hdf5(fname):
    print('Running PC ..')
    run_hdf5_alg(fname, 'pc')
    print('Running FGES ..')
    run_hdf5_alg(fname, 'fges')

def run_hdf5_alg(fname, alg):
    f = h5py.File(fname, 'r')
    raw_x = f['raw_x']
    raw_y = f['raw_y']
    # run 100?
    all_prec = 0
    all_recall = 0
    all_shd = 0
    ct = 0
    print('Runing for 20 data ..')
    # how many data
    num = raw_x.shape[0]
    indices = random.sample(range(num), 20)
    for i in indices:
        print('.', flush=True, end='')
        prec, recall, shd = run_one(alg, raw_x[i], raw_y[i])
        all_prec += prec
        all_recall += recall
        all_shd += shd
        ct += 1
    print()
    # calculate the average
    # FIXME performance not good, there must be something wrong
    print('prec:', all_prec / ct,
          'recall:', all_recall / ct,
          'shd:', all_shd / ct)

def main():
    os.chdir('julia/src')
    setup()
    run_hdf5('data/SF-10/d=10_k=1_gtype=SF_noise=Gaussian_mat=COR.hdf5')
    # teardown()

def run_one(alg, x, y):
    df = pd.DataFrame(x.transpose())
    assert df.shape[0] == 1000
    if alg == 'pc':
        # FIXME it seems that every run will reset the tetrad result, so I can
        # reuse the tetrad object
        tetrad.run(algoId = 'pc-all', dfs = df, testId = 'fisher-z-test', 
                   fasRule = 2, depth = 2, conflictRule = 1, concurrentFAS = True,
                   useMaxPOrientationHeuristic = True,
                   verbose = False)
    elif alg == 'fges':
        tetrad.run(algoId = 'fges', dfs = df, scoreId = 'sem-bic',
                   # priorKnowledge = prior,
                   maxDegree = -1, faithfulnessAssumed = True,
                   symmetricFirstStep = True, 
                   numberResampling = 5, resamplingEnsemble = 1,
                   addOriginalDataset = True,
                   verbose = False)
    else:
        assert False
        
    # convert it into adjacency_matrix
    pred = tetrad_get_adj(tetrad)
    # I probably don't want to calculate the precision, recall, shd here?
    # anyway, I'm calculating the metrics
    #
    # CAUTION FIXME y should be transposed too? But performance is still bad
    # UPDATE y should not be transposed
    adj_true = y
    return compute_metrics(pred, y)

def tetrad_get_adj(tetrad):
    # Seems that I have to extract it from string
    nodes = tetrad.getNodes()
    edges = tetrad.getEdges()
    d = len(nodes)
    res = np.zeros((d, d))
    for edge in edges:
        s = edge.split('[')[0].strip()
        a,arr,b = s.split(' ')
        a = int(a)
        b = int(b)
        if arr == '--':
            res[a,b] = 1
            res[b,a] = 1
        elif arr == '-->':
            res[a,b] = 1
        else:
            res[b,a] = 1
    return res
    

def show_svg(svg_str):
    folder = tempfile.mkdtemp()
    fname = os.path.join(folder, "a.svg")
    with open(fname, 'wb') as f:
        f.write(svg_str)
    print('#<Image: ' + fname + '>')
    

def test():
    f = h5py.File('data/SF-10/d=10_k=1_gtype=SF_noise=Gaussian_mat=COR.hdf5', 'r')
    f.keys()
    test_x = f['test_x']
    test_y = f['test_y']
    raw_x = f['raw_x']
    raw_y = f['raw_y']
    # this is numpy array
    test_x[:]
    test_x[1]
    # now the raw_x
    raw_x.shape
    raw_x[0].shape
    df = pd.DataFrame(raw_x[0].transpose())

    # I'll calculate for the first 100? Since they are actually randomly
    # generated, that should be fine. UPDATE I'll still randomly choose from it.
    dot_str = pc.tetradGraphToDot(tetrad.getTetradGraph())
    graphs = pydot.graph_from_dot_data(dot_str)
    svg_str = graphs[0].create_svg()

    show_svg(svg_str)
