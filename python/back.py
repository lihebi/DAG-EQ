
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