import tensorflow as tf
# from datetime import date, datetime, time
import datetime
import time
import re
import csv
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np



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
    m = re.match('.*/test-(.*)-d=(\d+)-ng=(\d+)(?:.0)?-N=(\d+)-.*', event_file)
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

def test_table():
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
        ax.plot(data[model][key], label=model)
    ax.set_ylabel(key)
    ax.set_title(key)
    ax.set_xticks(x)
    ax.set_xticklabels(ds)
    ax.legend()
    # plt.savefig('b.pdf')
    # plt.close()

def plot_bars(data, models, ds, ax, key):
    # fig, ax = plt.subplots()
    x = np.arange(len(ds)) * 2  # the label locations
    width = 0.35  # the width of the bars
    for i, model in enumerate(models):
        print(x)
        locs = x - width * len(models) / 2 + i * width + width / 2
        print(locs)
        ax.bar(locs, data[model][key], width, label=model)
        # I also want to plot the line
        # UPDATE this does not look good
        # ax.plot(locs, data[model][0], 'o-')
    ax.set_ylabel(key)
    ax.set_title(key)
    ax.set_xticks(x)
    ax.set_xticklabels(ds)
    ax.legend()
    # plt.savefig('a.pdf')
    # plt.close()

def plot_all(data, models, ds):
    fig, axs = plt.subplots(2,2, figsize=(10,10))
    plot_lines(data, models, ds, axs[0,0], 'precs')
    plot_lines(data, models, ds, axs[0,1], 'recalls')
    plot_bars(data, models, ds, axs[1,0], 'precs')
    plot_bars(data, models, ds, axs[1,1], 'recalls')
    plt.savefig('a.pdf')
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
def test_barplot():
    logdir = './final_logs'
    models = ['FC', 'FC-deep', 'EQ', 'EQ-deep']
    ds = [5,7,10,15,20,25,30]
    data = tfevent_to_plot_data(logdir, models, ds)
    # plot it
    plot_bars(data, models, ds)
    plot_lines(data, models, ds)
    plot_all(data, models, ds)

