import h5py
import os
import pandas as pd
import tempfile
import numpy as np
import random
import json
import time
import networkx as nx

import sys
sys.path.append("/home/hebi/git/reading/notears")
from notears.linear import notears_linear
from notears.nonlinear import NotearsMLP, notears_nonlinear

from baseline_common import compute_metrics
from baseline_cdt import run_CDT, run_RCC

from baseline_daggnn import dag_gnn
# from baseline_rlbic import rlbic

def read_hdf5(fname):
    f = h5py.File(fname, 'r')
    raw_x = f['raw_x']
    raw_y = f['raw_y']
    return raw_x, raw_y

def read_hdf5_iter(fname):
    f = h5py.File(fname, 'r')
    raw_x = f['raw_x']
    raw_y = f['raw_y']
    # get 20 in data frames
    num = raw_x.shape[0]
    indices = random.sample(range(num), 20)
    for i in indices:
        x = raw_x[i]
        y = raw_y[i]
        dfx = pd.DataFrame(x.transpose())
        yield dfx, y
    
def run_hdf5_alg(fname, alg):
    raw_x, raw_y = read_hdf5(fname)
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
    # calculate the average
    # FIXME performance not good, there must be something wrong
    print('prec:', all_prec / ct,
          'recall:', all_recall / ct,
          'shd:', all_shd / ct)

def run_one(alg, x, y):
    start = time.time()
    if alg in ['PC', 'SAM', 'CAM', 'GES']:
        mat = run_CDT(alg, x, y, False)
        prec, recall, shd = compute_metrics(mat, y)
        print('prec:', prec, 'recall:', recall, 'shd:', shd)
        return prec, recall, shd
    elif alg == 'notears':
        # np.numpy(x)
        # x.numpy()
        # np.array(pd.DataFrame(np.zeros((3,2)))).shape
        mat = notears_linear(np.array(x), lambda1=0, loss_type='l2')
        mat = (mat != 0).astype(np.int)
        # CAUTION here seems that I must do y.transpose()
        prec, recall, shd = compute_metrics(mat, y.transpose())
        print('prec:', prec, 'recall:', recall, 'shd:', shd)
        return prec, recall, shd
    elif alg == 'DAG-GNN':
        d = x.shape[1]
        # FIXME hyper-parameters
        mat = dag_gnn(d, np.array(x), y.transpose(), max_iter=5, num_epochs=100)
        # CAUTION I have to convert it into 1/0
        mat = (mat != 0).astype(np.int)
        prec, recall, shd = compute_metrics(mat, y.transpose())
        print('prec:', prec, 'recall:', recall, 'shd:', shd)
        return prec, recall, shd
    elif alg == 'RL-BIC':
        d = x.shape[1]
        try:
            mat = rlbic(d, np.array(x), y.transpose(),
                        lambda_iter_num=500, nb_epoch=2000)
            # FIXME the mat seems to be already int
            mat = (mat != 0).astype(np.int)
            # FIXME this seems to be wrong
            prec, recall, shd = compute_metrics(mat, y.transpose())
            print('prec:', prec, 'recall:', recall, 'shd:', shd)

            # And yes, no tranpose will result in worse performance
            # prec2, recall2, shd2 = compute_metrics(mat, y)
            # print('prec2:', prec2, 'recall2:', recall2, 'shd2:', shd2)

            return prec, recall, shd
        except KeyboardInterrupt as e:
            print('keyboard interrupt')
            exit(1)
    else:
        print('Unsupport alg', alg)
        assert(False)
    end = time.time()
    print('time: {:.3f}'.format(end-start))

def test():
    fname = 'data/SF-100/d=100_k=1_gtype=SF_noise=Gaussian_mat=COR.hdf5'
    fname = 'data/SF-200/d=200_k=1_gtype=SF_noise=Gaussian_mat=COR.hdf5'
    fname = 'data/SF-300/d=300_k=1_gtype=SF_noise=Gaussian_mat=COR.hdf5'
    fname = 'data/SF-400/d=400_k=1_gtype=SF_noise=Gaussian_mat=COR.hdf5'
    # run_many('GES', fname)
    it = read_hdf5_iter(fname)
    x, y = next(it)
    mat = run_CDT('GES', x, y, False)
    mat.astype(np.int)
    sum(sum(np.array(mat)))
    compute_metrics(mat, y)
    compute_metrics(np.array(mat), y)
    prec, recall, shd = compute_metrics(mat, y)
    print('prec:', prec, 'recall:', recall, 'shd:', shd)
    np.sum(mat != y)
    # number of predicted edges
    np.sum(mat == 1)
    # fp:
    np.sum(mat[y == 1] == 1)

def run_many(alg, fname):
    if alg in ['RCC-CLF', 'RCC-NN']:
        # test different methods
        raw_x, raw_y = read_hdf5(fname)
        if alg == 'RCC-CLF':
            return run_RCC(raw_x, raw_y, 'CLF')
        if alg == 'RCC-NN':
            # TODO save to result file
            return run_RCC(raw_x, raw_y, 'NN')
    else:
        it = read_hdf5_iter(fname)
        all_prec = 0
        all_recall = 0
        all_shd = 0
        ct = 0
        start = time.time()
        # FIXME testing only 10 graphs
        # DEBUG 6 for d=100
        for _ in range(5):
            x, y = next(it)
            prec, recall, shd = run_one(alg, x, y)
            ct += 1
            all_prec += prec
            all_recall += recall
            all_shd += shd
        end = time.time()
        return all_prec/ct, all_recall/ct, all_shd/ct, (end-start)/ct

def test():
    os.chdir('julia/src')
    raw_x, raw_y = read_hdf5('data/SF-10/d=10_k=1_gtype=SF_noise=Gaussian_mat=COR.hdf5')
    it = read_hdf5_iter('data/SF-10/d=10_k=1_gtype=SF_noise=Gaussian_mat=COR.hdf5')
    it = read_hdf5_iter('data/SF-20/d=20_k=1_gtype=SF_noise=Gaussian_mat=COR.hdf5')
    it = read_hdf5_iter('data/SF-50/d=50_k=1_gtype=SF_noise=Gaussian_mat=COR.hdf5')
    it = read_hdf5_iter('data/SF-100/d=100_k=1_gtype=SF_noise=Gaussian_mat=COR.hdf5')
    x, y = next(it)
    x
    y
    # also I want to measure the time
    timeit.timeit(1+1)
    run_one('PC', x, y)
    run_one('CAM', x, y)
    # FIXME this is pretty fast, maybe I don't need FGS
    run_one('GES', x, y)
    # TODO CAUTION this is very slow
    # run_one('SAM', x, y)
    run_one('notears', x, y)
    run_one('DAG-GNN', x, y)

    run_RCC(raw_x, raw_y, 'CLF')
    run_RCC(raw_x, raw_y, 'NN')

def load_results():
    fname = 'results/baseline.json'
    # I'm going to use object of:
    #
    # result_obj = {'data/xxx.hdf5': {'RCC-CLF': [prec, recall, shd],
    #                                 'RCC-NN': [prec, recall, shd],
    #                                 'PC': [prec, recall, shd]},
    #               'data/xxx': {}}
    #
    # when saving to csv file, I'll loop through the object
    #
    # UPDATE well, I don't have to use CSV. I can just use JSON
    if os.path.exists(fname):
        with open(fname, 'r') as fp:
            res = json.load(fp)
            return res
    else:
        return {}

def save_results(res):
    # TODO baseline results
    fname = 'results/baseline.json'
    with open(fname, 'w') as fp:
        json.dump(res, fp, indent=2)

def test():
    os.chdir('julia/src')

def main_large():
    for gtype in ['SF']:
        for d in [200, 300, 400]:
            fname = 'data/{}-{}/d={}_k=1_gtype={}_noise=Gaussian_mat=COR.hdf5'.format(gtype, d, d, gtype)
            print('== processing', fname, '..')
            # read baseline
            res = load_results()
            # FIXME notears seems to work too well
            for alg in ['GES']:
                print('-- running', alg, 'algorithm ..')
                if fname not in res:
                    res[fname] = {}
                if alg not in res[fname]:
                    prec, recall, shd, t = run_many(alg, fname)
                    res[fname][alg] = [prec, recall, shd, t]
                    print('-- testing result:', [prec, recall, shd, t])
                    print('-- writing ..')
                    save_results(res)

def test_RLBIC():
    gtype = 'SF'
    d = 10
    fname = 'data/{}-{}/d={}_k=1_gtype={}_noise=Gaussian_mat=COR.hdf5'.format(
        gtype, d, d, gtype)
    alg = 'RL-BIC'

    # prec, recall, shd, t = run_many(alg, fname)
    # print('-- testing result:', [prec, recall, shd, t])

    # running one
    it = read_hdf5_iter(fname)
    x, y = next(it)
    # prec, recall, shd = run_one(alg, x, y)

    # even inside for RL-BIC only
    d = x.shape[1]
    mat = rlbic(d, np.array(x), y.transpose(),
                lambda_iter_num=500, nb_epoch=2000)

    # compare mat with y
    compute_metrics(mat, y)
    # this seems to make sense the most
    compute_metrics(mat, y.transpose())
    # and this seems not necessary
    mat2 = (mat != 0).astype(np.int)
    compute_metrics(mat2, y)
    compute_metrics(mat2, y.transpose())

    # FIXME the mat seems to be already int
    mat = (mat != 0).astype(np.int)
    # FIXME this seems to be wrong
    prec, recall, shd = compute_metrics(mat, y.transpose())
    print('prec:', prec, 'recall:', recall, 'shd:', shd)

    prec2, recall2, shd2 = compute_metrics(mat, y)
    print('prec2:', prec2, 'recall2:', recall2, 'shd2:', shd2)

    return prec, recall, shd
    
    

def main():
    for gtype in ['SF', 'ER']:
        for d in [10, 20, 50, 100]:
            fname = 'data/{}-{}/d={}_k=1_gtype={}_noise=Gaussian_mat=COR_mec=Linear.hdf5'.format(gtype, d, d, gtype)
            print('== processing', fname, '..')
            # read baseline
            res = load_results()
            # FIXME notears seems to work too well
            for alg in ['PC', 'CAM', 'GES',
                        'RCC-CLF', 'RCC-NN',
                        'notears', 'DAG-GNN', 'RL-BIC']:
                print('-- running', alg, 'algorithm ..')
                if fname not in res:
                    res[fname] = {}
                if alg not in res[fname]:
                    # run RL-BIC only on d=10,20,50
                    if alg == 'RL-BIC' and d > 50: continue
                    prec, recall, shd, t = run_many(alg, fname)
                    res[fname][alg] = [prec, recall, shd, t]
                    print('-- testing result:', [prec, recall, shd, t])
                    print('-- writing ..')
                    save_results(res)
import csv
def table():
    # generate table for paper
    res = load_results()
    # generate csv directly
    # csv.write()
    with open('results/method.csv', 'w') as fp:
        writer = csv.writer(fp)
        # header
        writer.writerow(['model', 'prec', 'recall', 'shd', 'time'])
        for d in [10, 20, 50, 100]:
            name = "data/SF-{}/d={}_k=1_gtype=SF_noise=Gaussian_mat=COR.hdf5".format(d, d)
            methods = ['PC', 'GES', 'CAM',
                       'RCC-CLF', 'RCC-NN',
                       'notears', 'DAG-GNN']
            if d < 50:
                methods += ['RL-BIC']
            for method in methods:
                tmp = res[name][method]
                tmp = ['{:.1f}'.format(tmp[0] * 100),
                       '{:.1f}'.format(tmp[1] * 100),
                       tmp[2],
                       '{:.2f}'.format(tmp[3])]
                # tmp = list(map(lambda x: '{:.3f}'.format(x), tmp))
                tmp = [method] + tmp
                writer.writerow(tmp)
            writer.writerow([])

if __name__ == '__main__':
    main()
    # main_large()
