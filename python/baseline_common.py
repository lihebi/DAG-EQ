import numpy as np

def mydiv(a,b):
    if a == 0:
        return 0
    else:
        return a / b

def compute_metrics(pred, true):
    nnz = np.sum(pred != 0)
    nny = np.sum(true != 0)
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

