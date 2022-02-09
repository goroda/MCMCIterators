import itertools
import unittest
import pandas as pd
import numpy as np
import scipy.stats as scistats

# import os
# import sys
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../SamplingIterators')))

from MCMCIterators import samplers

class TestMonteCarloSampler(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cov = np.array([[1.0, 0.2],
                            [0.2, 0.8]])
        chol = np.linalg.cholesky(cls.cov)
        cls.mean = np.array([0.2, 0.5])
        cls.num_samples = 500000

        mh = samplers.MonteCarloSampler(lambda : cls.mean + np.dot(chol, np.random.randn(2)))
        cls.df = pd.DataFrame(itertools.islice(mh,cls.num_samples))
        
    def test_count(self):
        self.assertEqual(self.num_samples, self.df.count()[0])
        self.assertEqual(self.num_samples, self.df.count()[1])

    def test_mean(self):
        means = self.df.mean()
        self.assertAlmostEqual(self.mean[0], means[0], places=2)
        self.assertAlmostEqual(self.mean[1], means[1], places=2)
    
    def test_variance(self):
        variances = self.df.var()
        self.assertAlmostEqual(self.cov[0,0], variances[0], places=2)
        self.assertAlmostEqual(self.cov[1,1], variances[1], places=2)

    def test_covariance(self):
        cov = self.df.cov()
        self.assertAlmostEqual(self.cov[0,1], cov[0][1], places=2)
        
class TestRandomWalkGaussSampler(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.cov = np.array([[1.0, 0.2],
                             [0.2, 1.0]])
        cls.mean = np.array([0.2, 0.5])
        cls.num_samples = 50000
        logpdf_f = lambda x: scistats.multivariate_normal.logpdf(x,
                                                                 mean=cls.mean,
                                                                 cov=cls.cov)
        mh = samplers.RandomWalkGauss(logpdf_f, cls.mean, cls.cov*0.8)
        mh2 = itertools.starmap(lambda x,y,z: x, itertools.islice(mh, cls.num_samples))
        cls.df = pd.DataFrame(mh2)
        
    def test_count(self):
        self.assertEqual(self.num_samples, self.df.count()[0])
        self.assertEqual(self.num_samples, self.df.count()[1])

    def test_mean(self):
        means = self.df.mean()
        self.assertAlmostEqual(self.mean[0], means[0], places=1)
        self.assertAlmostEqual(self.mean[1], means[1], places=1)
    
    def test_variance(self):
        variances = self.df.var()
        self.assertAlmostEqual(self.cov[0,0], variances[0], places=1)
        self.assertAlmostEqual(self.cov[1,1], variances[1], places=1)

    def test_covariance(self):
        cov = self.df.cov()
        self.assertAlmostEqual(self.cov[0,1], cov[0][1], places=1)

class TestAdaptiveMetropolisGaussSampler(unittest.TestCase):

    @classmethod    
    def setUpClass(cls):
        cls.cov = np.array([[1.0, 0.2],
                             [0.2, 2.0]])
        cls.mean = np.array([0.2, 0.5])
        cls.num_samples = 100000
        logpdf_f = lambda x: scistats.multivariate_normal.logpdf(x,
                                                                 mean=cls.mean,
                                                                 cov=cls.cov)
        adapt_start = 10
        interval = 100
        mh = samplers.AdaptiveMetropolisGauss(logpdf_f, cls.mean, 0.2*np.eye(2),
                                              adapt_start = adapt_start,
                                              eps=1e-6, sd=2.4**2 / cls.cov.shape[0],
                                              interval=interval)
        mh2 = itertools.starmap(lambda x,y,z: x, itertools.islice(mh, cls.num_samples))
        cls.df = pd.DataFrame(mh2)
        
    def test_count(self):
        self.assertEqual(self.num_samples, self.df.count()[0])
        self.assertEqual(self.num_samples, self.df.count()[1])

    def test_mean(self):
        means = self.df.mean()
        self.assertAlmostEqual(self.mean[0], means[0], places=1)
        self.assertAlmostEqual(self.mean[1], means[1], places=1)
    
    def test_variance(self):
        variances = self.df.var()
        self.assertAlmostEqual(self.cov[0,0], variances[0], places=1)
        self.assertAlmostEqual(self.cov[1,1], variances[1], places=1)

    def test_covariance(self):
        cov = self.df.cov()
        self.assertAlmostEqual(self.cov[0,1], cov[0][1], places=1)        

class TestDelayedRejectionGaussSampler(unittest.TestCase):

    @classmethod    
    def setUpClass(cls):    
        cls.cov = np.array([[1.0, 0.2],
                             [0.2, 2.0]])
        cls.mean = np.array([0.2, 0.5])
        cls.num_samples = 200000
        logpdf_f = lambda x: scistats.multivariate_normal.logpdf(x,
                                                                 mean=cls.mean,
                                                                 cov=cls.cov)
        
        mh = samplers.DelayedRejectionGauss(logpdf_f, cls.mean, 0.2*np.eye(2), level_scale=1e-2)
        mh2 = itertools.starmap(lambda x,y,z: x, itertools.islice(mh, cls.num_samples))
        cls.df = pd.DataFrame(mh2)

    def test_count(self):
        self.assertEqual(self.num_samples, self.df.count()[0])
        self.assertEqual(self.num_samples, self.df.count()[1])

    def test_mean(self):
        means = self.df.mean()
        self.assertAlmostEqual(self.mean[0], means[0], places=1)
        self.assertAlmostEqual(self.mean[1], means[1], places=1)

    def test_variance(self):
        variances = self.df.var()
        self.assertAlmostEqual(self.cov[0,0], variances[0], places=1)
        self.assertAlmostEqual(self.cov[1,1], variances[1], places=1)
        
    def test_covariance(self):
        cov = self.df.cov()
        self.assertAlmostEqual(self.cov[0,1], cov[0][1], places=1)


class TestDelayedRejectionAdaptiveMetropolisGaussSampler(unittest.TestCase):

    @classmethod    
    def setUpClass(cls):    
        cls.cov = np.array([[1.0, 0.2],
                             [0.2, 2.0]])
        cls.mean = np.array([0.2, 0.5])
        cls.num_samples = 100000
        logpdf_f = lambda x: scistats.multivariate_normal.logpdf(x,
                                                                 mean=cls.mean,
                                                                 cov=cls.cov)
        
        mh = samplers.DelayedRejectionAdaptiveMetropolis(logpdf_f, cls.mean, 0.2*np.eye(2),
                                                         adapt_start=10,
                                                         eps=1e-6, sd=2.4**2 / cls.cov.shape[0],
                                                         interval=10,
                                                         level_scale=1e-2)
        mh2 = itertools.starmap(lambda x,y,z: x, itertools.islice(mh, cls.num_samples))
        cls.df = pd.DataFrame(mh2)

    def test_count(self):
        self.assertEqual(self.num_samples, self.df.count()[0])
        self.assertEqual(self.num_samples, self.df.count()[1])

    def test_mean(self):
        means = self.df.mean()
        self.assertAlmostEqual(self.mean[0], means[0], places=1)
        self.assertAlmostEqual(self.mean[1], means[1], places=1)

    def test_variance(self):
        variances = self.df.var()
        self.assertAlmostEqual(self.cov[0,0], variances[0], places=1)
        self.assertAlmostEqual(self.cov[1,1], variances[1], places=1)
        
    def test_covariance(self):
        cov = self.df.cov()
        self.assertAlmostEqual(self.cov[0,1], cov[0][1], places=1)        
        
if __name__ == '__main__':
    unittest.main(verbosity=2)
