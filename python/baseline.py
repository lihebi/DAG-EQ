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
# Add to sys.path

from baseline_common import compute_metrics
from baseline_cdt import run_CDT, run_RCC
from bl_gob import run_Gob

sys.path.append(os.path.abspath("./notears"))
from notears.linear import notears_linear
from notears.nonlinear import NotearsMLP, notears_nonlinear

sys.path.append(os.path.abspath("./DAG-GNN/src"))
from baseline_daggnn import dag_gnn

# sys.path.append(os.path.expanduser("~/trustworthyAI/Causal_Structure_Learning/Causal_Discovery_RL/src/"))
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
    if alg in ['PC', 'SAM', 'CAM', 'GES']:
        mat = run_CDT(alg, x, y, False)
        prec, recall, shd = compute_metrics(mat, y, isPDAG=True)
        # print('prec:', prec, 'recall:', recall, 'shd:', shd)
        return prec, recall, shd
    elif alg == 'gob':
        mat = run_Gob(x, y)
        prec, recall, shd = compute_metrics(mat, y, isPDAG=True)
        # print('prec:', prec, 'recall:', recall, 'shd:', shd)
        return prec, recall, shd
    elif alg == 'notears':
        # np.numpy(x)
        # x.numpy()
        # np.array(pd.DataFrame(np.zeros((3,2)))).shape
        #
        # FIXME should I use lambda1=0.1?
        mat = notears_linear(np.array(x), lambda1=0.1, loss_type='l2')
        mat = (mat != 0).astype(np.int)
        # CAUTION here seems that I must do y.transpose()
        prec, recall, shd = compute_metrics(mat, y.transpose())
        # print('prec:', prec, 'recall:', recall, 'shd:', shd)
        return prec, recall, shd
    elif alg == 'DAG-GNN':
        d = x.shape[1]
        # FIXME hyper-parameters
        mat = dag_gnn(d, np.array(x), y.transpose(), max_iter=5, num_epochs=100)
        # CAUTION I have to convert it into 1/0
        mat = (mat != 0).astype(np.int)
        prec, recall, shd = compute_metrics(mat, y.transpose())
        # print('prec:', prec, 'recall:', recall, 'shd:', shd)
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
            # print('prec:', prec, 'recall:', recall, 'shd:', shd)

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


from multiprocessing import Process, Queue
from multiprocessing import Pool, TimeoutError
from multiprocessing.pool import ThreadPool

def f(data):
    alg, x, y = data
    print('running one run for', alg, '..')
    start = time.time()
    # FIXME testing only 10 graphs
    # DEBUG 6 for d=100
    prec, recall, shd = run_one(alg, x, y)
    end = time.time()
    oneresult = [prec, recall, shd, end-start]
    print('finished one run, the one result:', oneresult)
    return oneresult

def timeout_f(data):
    with ThreadPool(processes=1) as pool:
        res = pool.apply_async(f, (data,))
        try:
            # FIXME hardcoded timeout
            return res.get(timeout=5000)
        except TimeoutError:
            print('timeout, returning none')
            return None

def run_many_parallel(alg, fname, nruns=5):
    # run all the runs in parallel
    assert nruns <= 20
    # q = Queue()
    # for i in range(nruns):
    #     p = Process(target=f, args=(q,))
    #     p.start()
    #     p.join()
    # FIXME what if nruns larger than nprocesses?
    with Pool(processes=1) as pool:
        # get the data
        it = read_hdf5_iter(fname)
        # get the nruns data
        data = [[alg] + list(next(it)) for _ in range(nruns)]
        print('spawn out workers on the task ..')
        # Option 1
        # results = pool.map(f, data)
        #
        # Option 2
        # DEBUG timeout_f
        # FIXME handle the None values
        results = pool.map(timeout_f, data)
        #
        # Option 3
        # async
        # async_results = [pool.apply_async(f, (d,)) for d in data]
        # results = []
        # for a in async_results:
        #     try:
        #         # FIXME the first one get up to 100s, the second one get up to 200s
        #         results.append(a.get(timeout=100))
        #     except TimeoutError:
        #         # do not append
        #         print("warning: timeout, result skipped")
        #         pass
        print('results:', results)
        # remove None
        results = [x for x in results if x]
        # check how many results left
        print("CAUTION: No. of results left", len(results))
        # FIXME return average of results
        avgres = np.mean(results, axis=0).tolist()
        print('avg:', avgres)
        return avgres

def run_many(alg, fname, nruns=5):
    assert os.path.exists(fname), fname
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
        
        # FIXME testing only 10 graphs
        # DEBUG 6 for d=100
        results = []
        for i in range(nruns):
            print("Run No.{}".format(i+1))
            x, y = next(it)
            start = time.time()
            prec, recall, shd = run_one(alg, x, y)
            end = time.time()
            oneresult = [prec, recall, shd, end-start]
            print('the one result:', oneresult)
            results.append(oneresult)
        avgres = np.mean(results, axis=0).tolist()
        print('avg:', avgres)
        return avgres

def load_results(fname = 'results/baseline.json'):
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


# CAUTION fname must be kept consistent with load_results!!
def save_results(res, fname = 'results/baseline.json'):
    # TODO baseline results
    if not os.path.exists('results'):
        os.makedirs('results')
    with open(fname, 'w') as fp:
        json.dump(res, fp, indent=2)

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
                    # prec, recall, shd, t
                    oneresult = run_many(alg, fname)
                    res[fname][alg] = oneresult
                    print('-- testing result:', oneresult)
                    print('-- writing ..')
                    save_results(res)

    
# get dataset file name
def get_dataset_fname(datadir, d, gtype, mat='CH3'):
    # FIXME since I'm using raw for the baselines, I should be fine using the previous results.
    # But the previous result is for SF only. If I want to report the average of
    # SF/ER, I need to run it again.
    #
    # FIXME this is ugly
    if d >= 300:
        ng = 300
        N = 1
    elif d >= 50:
        ng = 1000
        N = 1
    else:
        ng = 3000
        N = 3
    return '{}/{}-{}-ng={}-1234/d={}_k=1_gtype={}_noise=Gaussian_mat={}_mec=Linear_ng={}_N={}.hdf5'.format(datadir, gtype, d, ng, d, gtype, mat, ng, N)


def main_parallel(ds, algs, nruns):
    for d in ds:
        for gtype in ['SF', 'ER']:
            fname = get_dataset_fname("../notebooks/data", d, gtype)
            print('== processing', fname, '..')
            # read baseline
            # DEBUG using a testing json file
            res = load_results()
            # FIXME notears seems to work too well
            for alg in algs:
                print('-- running', alg, 'algorithm ..')
                if fname not in res:
                    res[fname] = {}
                if alg not in res[fname]:
                    # run RL-BIC only on d=10,20,50
                    if alg == 'RL-BIC' and d > 50: continue
                    # prec, recall, shd, t
                    oneresult = run_many_parallel(alg, fname, nruns=nruns)
                    res[fname][alg] = oneresult
                    print('-- testing result:', oneresult)
                    print('-- writing ..')
                    save_results(res)

def main(ds, algs, nruns):
    for d in ds:
        for gtype in ['SF', 'ER']:
            fname = get_dataset_fname("../notebooks/data", d, gtype)
            print('== processing', fname, '..')
            # read baseline
            res = load_results()
            # FIXME notears seems to work too well
            for alg in algs:
                print('-- running', alg, 'algorithm ..')
                if fname not in res:
                    res[fname] = {}
                if alg not in res[fname]:
                    # run RL-BIC only on d=10,20,50
                    if alg == 'RL-BIC' and d > 50: continue
                    # prec, recall, shd, t
                    oneresult = run_many(alg, fname, nruns=nruns)
                    res[fname][alg] = oneresult
                    print('-- testing result:', oneresult)
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
            # CAUTION fixed "SF" here
            #
            # FIXME reading previous results
            name = get_dataset_fname_old(d, 'SF')
            methods = ['PC', 'GES', 
                       # CAM is a little slow
#                        'CAM',
                       'RCC-CLF', 'RCC-NN',
                       'notears', 'DAG-GNN'
                       ]
#             if d < 50:
#                 methods += ['RL-BIC']
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
    main([10, 20, 50, 100], ['PC', 
                        'GES',
                        'CAM',
                        # 'RCC-CLF', 'RCC-NN',
                        'notears',
                        # 'DAG-GNN',
#                         'RL-BIC'
                       ])
    # main_large()
