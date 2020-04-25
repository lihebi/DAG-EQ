import tensorflow as tf
# from datetime import date, datetime, time
import datetime
import time
import re
import csv
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
# for smoothing
from scipy.interpolate import make_interp_spline, BSpline
from scipy.ndimage.filters import gaussian_filter1d


# FIXME smoothing
def read_tfevent_last_values(event_file):
    # read event file for results
    it = tf.compat.v1.train.summary_iterator(event_file)

    # Can I just peek the last element in the iterator?
    # - https://stackoverflow.com/questions/2138873/
    # - PEP 448
    #
    # UPDATE: well, this is one value actually, I need several "last values"
    #
    # *_, last = it

    res = {}

    # First, the exp setting
    #
    # HACK (?:.0)? is for 10000.0, make it consistent
    m = re.match('.*/test-NEW-(.*)-d=(\d+)-ng=(\d+)(?:.0)?-N=(\d+)-.*', event_file)
    assert m
    m.groups()
    model, d, ng, N = m.groups()
    # HACK naming consistency
    # FIXME the d=10 case is strange: it does not match -dropout version
    if model == 'EQ-deep-dropout': model = 'EQ-deep'
    # TODO load into csv
    res['model'] = model
    res['d'] = int(d)
    res['ng'] = int(ng)
    res['N'] = int(N)

    # read the data
    first = next(it)
    # the first seems to be a placeholder
    assert first.step == 0
    assert len(first.summary.value) == 0
    # but I need  the time
    start_wall = first.wall_time
    res['start_time'] = start_wall
    for e in it:
        res['wall_time'] = e.wall_time
        res['step'] = e.step
        # seems each summary contains exactly one value (for scalar logs)
        assert len(e.summary.value) == 1
        for v in e.summary.value:
            res[v.tag] = v.simple_value
    res['time'] = res['wall_time'] - start_wall
    return res

def seconds_to_str(delta):
    hour = int(delta) // 3600
    minute = int(delta) % 3600 // 60
    return "{}:{}".format(hour, minute)
def tfevent_to_table_row(event_file):
    # I need:
    # 1. the value and tag (easy)
    # 2. the relative time
    row = read_tfevent_last_values(event_file)
    # update fields
    row['loss'] = '{:.3f}'.format(row['loss'])
    row['prec'] = '{:.3f}'.format(row['graph/prec'])
    row['recall'] = '{:.3f}'.format(row['graph/recall'])
    row['fpr'] = row['graph/fpr']
    row['fdr'] = row['graph/fdr']
    row['tpr'] = row['graph/tpr']
    # TODO get this into a string
    row['time (H:M)'] = seconds_to_str(row['time'])
    row['step'] = '{}k'.format(row['step'] // 1000)
    return row

def gen_table():
    # generate a table
    # final loss dir
    logdir = './final_logs'
    os.listdir(logdir)

    rows = []
    for folder in os.listdir(logdir):
        tmp = os.path.join(logdir, folder)
        # assume only one file in each folder
        assert len(os.listdir(tmp)) == 1
        event_file = os.path.join(tmp, os.listdir(tmp)[0])
        row = tfevent_to_table_row(event_file)
        rows.append(row)
    # sort rows
    # 1. model
    # 2. d
    rows = sorted(rows, key=lambda x: (x['model'], x['d']))
    # process rows
    with open('out.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        keys = ['model', 'd', 'ng', 'N', 'loss', 'prec', 'recall', 'time (H:M)', 'step']
        writer.writerow(keys)
        for row in rows:
            writer.writerow([row[key] for key in keys])
    # TODO load csv into paper

def plot_lines(data, models, ds, ax, key):
    # fig, ax = plt.subplots()
    for model in models:
        ax.plot(data[model][key], 'o-', label=model)
    ax.set_ylabel(key)
    ax.set_xlabel('graph size')
    # ax.set_title(key)
    # ax.set_xticks(x)
    ax.set_xticklabels(ds)
    ax.legend()

def plot_bars(data, models, ds, ax, key):
    # fig, ax = plt.subplots()
    x = np.arange(len(ds)) * 2  # the label locations
    width = 0.35  # the width of the bars
    for i, model in enumerate(models):
        locs = x - width * len(models) / 2 + i * width + width / 2
        ax.bar(locs, data[model][key], width, label=model)
    ax.set_ylabel(key)
    ax.set_xlabel('graph size')
    # ax.set_title(key)
    ax.set_xticks(x)
    ax.set_xticklabels(ds)
    ax.legend()

def plot_all(data, models, ds):
    fig, axs = plt.subplots(1,2, figsize=(10,5))
    plot_bars(data, models, ds, axs[0], 'precs')
    plot_bars(data, models, ds, axs[1], 'recalls')
    # plot_lines(data, models, ds, axs[1,0], 'precs')
    # plot_lines(data, models, ds, axs[1,1], 'recalls')
    plt.savefig('results/bar.pdf')
    plt.close()

def tfevent_to_plot_data(logdir, models, ds):
    os.listdir(logdir)
    # 1. read all values
    rows = []
    for folder in os.listdir(logdir):
        tmp = os.path.join(logdir, folder)
        # assume only one file in each folder
        assert len(os.listdir(tmp)) == 1
        event_file = os.path.join(tmp, os.listdir(tmp)[0])
        row = read_tfevent_last_values(event_file)
        rows.append(row)
    # for each row, group by model
    data = {}
    for model in models:
        # In each model, sort by d, fill in missing value
        prec = defaultdict(lambda: 0)
        recall = defaultdict(lambda: 0)
        for row in rows:
            if row['model'] == model:
                prec[row['d']] = row['graph/prec']
                recall[row['d']] = row['graph/recall']
        precs = [prec[d] for d in ds]
        recalls = [recall[d] for d in ds]
        # use precs and recalls
        data[model] = {}
        data[model]['precs'] = precs
        data[model]['recalls'] = recalls
    return data
def gen_barplot():
    logdir = './final_logs'
    models = ['EQ', 'EQ-deep', 'FC', 'FC-deep']
    ds = [5,10,15,20,25,30]
    data = tfevent_to_plot_data(logdir, models, ds)
    # plot it
    # plot_bars(data, models, ds)
    # plot_lines(data, models, ds)
    plot_all(data, models, ds)

def read_tfevent_process_values(event_file):
    # read event file for results
    it = tf.compat.v1.train.summary_iterator(event_file)
    res = {}

    # First, the exp setting
    #
    # HACK (?:.0)? is for 10000.0, make it consistent
    #
    # CAUTION hard-coded NEW- prefix
    m = re.match('.*/test-NEW-(.*)-d=(\d+)-ng=(\d+)(?:.0)?-N=(\d+)-.*', event_file)
    assert m
    m.groups()
    model, d, ng, N = m.groups()
    # FIXME the d=10 case is strange: it does not match -dropout version
    # TODO load into csv
    res['model'] = model
    res['d'] = int(d)
    res['ng'] = int(ng)
    res['N'] = int(N)
    steps = []
    # FIXME for train data, only losses are available. I'm currently not
    # plotting loss, only precs and recalls, and not considering training data
    #
    # losses = []
    precs = []
    recalls = []

    # read the data
    first = next(it)
    # the first seems to be a placeholder
    assert first.step == 0
    assert len(first.summary.value) == 0
    # but I need  the time
    start_wall = first.wall_time
    for e in it:
        # seems each summary contains exactly one value (for scalar logs)
        assert len(e.summary.value) == 1
        v = e.summary.value[0]
        if v.tag == 'graph/prec':
            # NOTE: steps only push here, not on recall
            steps.append(e.step)
            precs.append(v.simple_value)
        elif v.tag == 'graph/recall':
            recalls.append(v.simple_value)
        else:
            pass
    res['steps'] = np.array(steps)
    res['precs'] = np.array(precs)
    res['recalls'] = np.array(recalls)
    assert len(res['steps']) == len(res['precs'])
    assert len(res['steps']) == len(res['recalls'])
    return res

def tfevent_to_process_data(logdir):
    os.listdir(logdir)
    # 1. read all values
    rows = []
    for folder in os.listdir(logdir):
        tmp = os.path.join(logdir, folder)
        # assume only one file in each folder
        assert len(os.listdir(tmp)) == 1
        event_file = os.path.join(tmp, os.listdir(tmp)[0])
        row = read_tfevent_process_values(event_file)
        rows.append(row)
    # plot one figure for EQ and EQ-deep, one figure for FC and FC-deep, because
    # they have different X-axis
    EQ_rows = [row for row in rows if row['model'] in ['EQ', 'EQ-deep']]
    FC_rows = [row for row in rows if row['model'] in ['FC', 'FC-deep']]
    return EQ_rows, FC_rows

# consistent coloring for the same d
def my_color_map(model, d):
    res = {5: 'r',
           10: 'g',
           15: 'b',
           20: 'y',
           25: 'c',
           30: 'm'}[d]
    res += {'EQ': '--',
            'FC': '--',
            'EQ-deep': '-',
            'FC-deep': '-'}[model]
    return res

def plot_process_sub(rows, ax, which):
    # sort rows based on its last value
    rows = sorted(rows, key=lambda x: x[which][-1], reverse=True)

    for row in rows:
        # print(row['model'], row['d'], which)
        # print('{}-{}-{}'.format(row['model'], row['d'], which))

        y = row[which]
        x = row['steps']
        # smoothing
        # 300 represents number of points to make between T.min and T.max
        # ratio = 0.2 if 'EQ' in row['model'] else 1
        # xnew = np.linspace(x.min(), x.max(), int(len(x) * ratio))
        xnew = np.linspace(x.min(), x.max(), 20)
        spl = make_interp_spline(x, y, k=3)  # type: BSpline
        ynew = spl(xnew)

        # this seems to look smoother
        xnew = x
        if 'EQ' in row['model']:
            ynew = gaussian_filter1d(y, sigma=2)
            # ynew = y
        else:
            ynew = y

        ax.plot(xnew, ynew, my_color_map(row['model'], row['d']),
                label='{}-{}'.format(row['model'], row['d']))
        ax.set_ylabel(which)
        ax.set_xlabel('steps')
        # ax.set_title(which)
        ax.set_ylim(0,1)
        ax.xaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda x, p: '{}k'.format(int(x/1000))))

        # ax.set_xticks(x)
        # ax.set_xticklabels(ds)
        ax.legend(fontsize='medium', ncol=2)

def plot_process(EQ_rows, FC_rows):
    fig, axs = plt.subplots(2,2, figsize=(10,10))
    plot_process_sub(EQ_rows, axs[0,0], 'precs')
    plot_process_sub(EQ_rows, axs[1,0], 'recalls')
    plot_process_sub(FC_rows, axs[0,1], 'precs')
    plot_process_sub(FC_rows, axs[1,1], 'recalls')
    plt.savefig('results/process.pdf')
    plt.close()

def gen_plot_train_process():
    logdir = './final_logs'
    # this should be the entire training process
    EQ_rows, FC_rows = tfevent_to_process_data(logdir)
    plot_process(EQ_rows, FC_rows)

def gen_universal_plot():
    xs = []
    precs = []
    recalls = []
    with open("results/sf.csv") as f:
        reader = csv.reader(f)
        for row in reader:
            xs.append(int(row[0]))
            precs.append(round(float(row[1]), ndigits=3))
            recalls.append(round(float(row[2]), ndigits=3))
    # do with xs, precs, recalls
    xs
    precs
    recalls
    # print each
    print('xs', xs)
    print('precs', precs)
    print('recalls', recalls)
    # bar plot
    fig = plt.figure(dpi=600)

    width = 0.35
    index = np.arange(len(xs))

    plt.bar(index-width/2, precs, width, label="prec")
    plt.bar(index+width/2, recalls, width, label="recall")
    # plt.xticks(index+width/2, xs)
    plt.xticks(index, xs)
    plt.xlabel('test graph size')
    plt.title('universal model')
    plt.legend()
    plt.savefig("results/universal.pdf")
    plt.close()

def main():
    gen_barplot()
    gen_plot_train_process()
    gen_table()
    gen_universal_plot()
