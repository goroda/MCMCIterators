"""Compare MH, AM, DR, and DRAM on a Banana Function"""


import numpy as np
import pandas as pd
from pandas.plotting import autocorrelation_plot
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import itertools

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../SamplingIterators')))

import samplers, utils

def banana(x):

    if x.ndim == 1:
        x = x[np.newaxis, :]
    N, d = x.shape
    x1p = x[:, 0]
    x2p = x[:, 1] + (np.square(x[:, 0]) + 1)
    xp = np.concatenate((x1p[:, np.newaxis], x2p[:,np.newaxis ]),
                        axis=1)
    sigma = np.array ([[1, 0.9], [0.9, 1]])
    mu = np.array([0, 0])
    preexp = 1.0 / (2.0 * np.pi)**(d/2) / np.linalg.det(sigma)**0.5
    diff = xp - np.tile(mu[np.newaxis , :], (N, 1))
    sol = np.linalg.solve(sigma , diff.T)
    inexp = np.einsum("ij ,ij ->j",diff.T, sol)
    return np.log(preexp) - 0.5 * inexp


if __name__ == "__main__":

    print("Hello World")

    x_rand = np.random.randn(2)
    x0, cov0 = utils.laplace_approx(x_rand, banana)
    cov_opt = 2.4**2/2 * cov0
    nsamples = 20000
    
    mh = samplers.RandomWalkGauss(banana, x0, cov_opt)

    results = itertools.islice(mh, nsamples)
    samples = itertools.starmap(lambda x,y,z: x, results)
    
    df = pd.DataFrame(samples, columns=['x1', 'x2'])

    # df.plot()
    # autocorrelation_plot(df['x1'])
    plt.savefig('samples.pdf')
    # plt.show()
    
    
