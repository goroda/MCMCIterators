import itertools
import unittest
import pandas as pd
import numpy as np
import scipy.stats as scistats

# import os
# import sys
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../SamplingIterators')))

from MCMCIterators import samplers

class TestSampleCovariance(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        cls.samples = np.random.randn(1000, 5)
        cls.target_cov = np.cov(cls.samples, rowvar=False)
        cls.target_mean = np.mean(cls.samples, axis=0)


    def test_cov_computation(self):

        sd = 1.23416
        eps = 0.123
        mh = samplers.AdaptiveMetropolisGauss(lambda x: x,
                                              self.samples[0, :],
                                              np.eye(5),
                                              adapt_start=None,
                                              eps=eps,
                                              sd=sd,
                                              interval=None)

        nsamples = self.samples.shape[0]
        for ii in range(1, nsamples):
            mh.k += 1
            mh.update_cov_and_mean(self.samples[ii, :])

        # print(f"Mean = {mh.mean}")
        # print(f"Target mean = {self.target_mean}")

        target_cov = sd*self.target_cov + sd * eps * np.eye(5)        
        # print(f"Cov = {mh.S}")
        # print(f"Tcov = {target_cov}")

        diff_mean = np.linalg.norm(self.target_mean - mh.mean)
        diff_cov = np.linalg.norm(target_cov - mh.S)
        self.assertAlmostEqual(diff_mean, 0, places=10)
        self.assertAlmostEqual(diff_cov, 0, places=10)
        
        # self.assertEqual(1,1)

if __name__ == '__main__':
    unittest.main(verbosity=2)

