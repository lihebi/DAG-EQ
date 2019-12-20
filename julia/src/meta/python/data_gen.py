# import sys
# sys.path.insert(0, '../..')

import numpy as np
import torch
import matplotlib.pyplot as plt
import tempfile

from data_utils import RandomSplineSCM

scm = RandomSplineSCM(input_noise=False, output_noise=True,
    span=8., num_anchors=8, order=2, range_scale=1.)
num_points = 1000

def myplot():

    plt.figure(figsize=(9, 5))
    ax = plt.subplot(1, 1, 1)
    mus = [0, -4., 4.]
    colors = ['C0', 'C3', 'C2']
    labels = ['Training', r'Transfer ($\mu = -4$)', r'Transfer ($\mu = +4$)']

    for i, (mu, color, label) in enumerate(zip(mus, colors, labels)):
        X = mu + 2 * torch.randn((1000, 1))
        ax.scatter(X.squeeze(1).numpy(), scm(X).squeeze(1).numpy(),
                   color=color, marker='+', alpha=0.5, label=label, zorder=2 - i)

    ax.tick_params(axis='both', which='major', labelsize=13)
    ax.legend(loc=4, prop={'size': 13})
    ax.set_xlabel('A', fontsize=14)
    ax.set_ylabel('B', fontsize=14)

    # plt.show()
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    plt.savefig(tmp)
    print(tmp.name)
    print('#<Image: ' + tmp.name + '>')
    plt.close()
    # plt.savefig("/tmp/a.png")
    # print('#<Image: ' + '/tmp/a.png' + '>')

myplot()
