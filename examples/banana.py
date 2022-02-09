"""Compare MH, AM, DR, and DRAM on a Banana Function."""

import numpy as np
import pandas as pd

# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import itertools

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                '../SamplingIterators')))

import samplers
import utils


def banana(x):
    """Evaluate a banana shaped distribution."""
    if x.ndim == 1:
        x = x[np.newaxis, :]
    N, d = x.shape
    x1p = x[:, 0]
    x2p = x[:, 1] + (np.square(x[:, 0]) + 1)
    xp = np.concatenate((x1p[:, np.newaxis], x2p[:, np.newaxis]), axis=1)
    sigma = np.array([[1, 0.9], [0.9, 1]])
    mu = np.array([0, 0])
    preexp = 1.0 / (2.0 * np.pi)**(d/2) / np.linalg.det(sigma)**0.5
    diff = xp - np.tile(mu[np.newaxis, :], (N, 1))
    sol = np.linalg.solve(sigma, diff.T)
    inexp = np.einsum("ij ,ij ->j", diff.T, sol)
    return np.log(preexp) - 0.5 * inexp


if __name__ == "__main__":

    # If using the Laplace Approximation
    # x_rand = np.random.randn(2)
    # x0, cov0 = utils.laplace_approx(x_rand, banana)

    # Results from Laplace should be close to below
    x0 = np.array([0, -1.0])
    cov0 = np.array([[1.0, 0.9],
                     [0.9, 1.0]])

    # Example usage
    sampler = samplers.RandomWalkGauss(banana, x0, 2*cov0)

    # only consider 10 samples
    finite_sampler = itertools.islice(sampler, 10)
    for sample, logpdf, accepted_bool in finite_sampler:
        print(f"Sample: {sample}")
        print(f"\t Logpdf: {logpdf}")
        print(f"\t Accepted? -> {accepted_bool}")
        print("\n")

    # Comparison of algorithms with all algorithmic options set
    sampler_names = ['MH', 'Adaptive', 'Delayed Rejection', 'DRAM']
    sampler_types = [samplers.RandomWalkGauss(banana, x0, 2*cov0),
                     samplers.AdaptiveMetropolisGauss(banana, x0, 2*cov0,
                                                      adapt_start=10,
                                                      eps=1e-8,
                                                      sd=None,
                                                      interval=1),
                     samplers.DelayedRejectionGauss(banana, x0, 2*cov0,
                                                    level_scale=1e-1),
                     samplers.DelayedRejectionAdaptiveMetropolis(
                         banana, x0, 2*cov0,
                         adapt_start=10,
                         eps=1e-8,
                         sd=None,
                         interval=1,
                         level_scale=1e-1)]

    nsamples = 100000
    print(f"Now generating {nsamples} samples from all algorithms. Please wait...")
    burnin = 20000
    maxlag = 300
    for name, sampler in zip(sampler_names, sampler_types):

        # Here we build a couple of iterators on top of one
        # another to extract a fixed number of samples

        # truncate iterator to nsamples
        results = itertools.islice(sampler, nsamples)
        # Only extract samples, ignoring second two outputs 
        samples = itertools.starmap(lambda x, y, z: x, results)

        df = pd.DataFrame(samples, columns=['x1', 'x2'])

        corr = utils.auto_correlation(df['x1'].iloc[burnin:].to_numpy())
        plt.figure(1)
        plt.plot(corr[:maxlag])

        corr = utils.auto_correlation(df['x2'].iloc[burnin:].to_numpy())
        plt.figure(2)
        plt.plot(corr[:maxlag])

    plt.figure(1)
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation x1')
    plt.legend(sampler_names)
    plt.xlim([0, maxlag])
    # plt.savefig('autocorr1.pdf')

    plt.figure(2)
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation x2')
    plt.legend(sampler_names)
    plt.xlim([0, maxlag])
    # plt.savefig('autocorr2.pdf')

    plt.show()

