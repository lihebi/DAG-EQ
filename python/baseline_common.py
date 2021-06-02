import numpy as np

def mydiv(a,b):
    if a == 0:
        return 0
    else:
        return a / b

def compute_metrics(pred, true, isPDAG=False):
    true = true.copy()
    nnz = np.sum(pred != 0)
    nny = np.sum(true != 0)
    # if it is PDAG, then
    # - find all undirected edges
    # - if either edge present in true graph, MODIFY the true graph to be 1,1,
    #   else to 0,0
    if isPDAG:
        # find all undirected edges
        for i in range(len(pred)):
            for j in range(i):
                if pred[i,j] == pred[j,i] == 1:
                    print("DEBUG undirected edge found", i, j)
                    # FIXME mutating the true matrix in place
                    if true[i,j] == 1 or true[j,i] == 1:
                        true[i,j] = true[j,i] =1
                    else:
                        true[i,j] = true[j,i] = 0
    tp = np.sum(pred[true == 1] == 1)
    fp = np.sum(pred[true == 0] == 1)
    tt = np.sum(true == 1)
    ff = np.sum(true == 0)
    # print(tp, fp, tt, ff)

    prec = mydiv(tp, tp+fp)
    recall = mydiv(tp, tt)

    tpr = mydiv(tp, tt)
    fpr = mydiv(fp, ff)
    fdr = mydiv(fp, np.sum(pred == 1))

    shd = np.sum(true != pred)
    # print('prec:', prec,
    #       'recall:', recall,
    #       'shd:', shd)
    return prec, recall, shd

