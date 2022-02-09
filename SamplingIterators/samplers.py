import abc
import numpy as np
import copy

# StopIteration # <-- To stop an iteration inside next

def sample_gauss(mean, cov_chol):
    return mean + np.dot(cov_chol, np.random.randn(mean.shape[0]))

class SequentialSampler(abc.ABC):
    
    @abc.abstractmethod
    def update_state(self, new_sample):
        """Update the state with the new sample
        """
        raise NotImplementedError("Should implement update_state")
    
    @abc.abstractmethod
    def step(self):
        """Produce the next sample
        """
        raise NotImplementedError("Should implement step")
    
    def __iter__(self):
        return self

    def __next__(self):
        new_step = self.step()
        self.update_state(new_step)
        return new_step

class MonteCarloSampler(SequentialSampler):
    """An example of how Monte Carlo can be implemented as 
    a SequentialSampler. Obviously you would not want to do this,
    because Monte Carlo can be vectorized and done in parallel."""
    def __init__(self, sampler):
        self.sampler = sampler
    
    def update_state(self, new_sample):
        pass

    def step(self):
        return self.sampler()

class MetropolisHastings(SequentialSampler, abc.ABC):
    """Metropolis Hastings Sampler. 

    This is an abstract class that specifies the required
    functions that must be filled in for a concrete realization

    In particular, the concrete classes will need to implement the following methods

    process_new_sample
    propose
    step
    """
    
    def __init__(self, logpdf, initial_sample):
        self.current_sample = initial_sample
        self.current_logpdf = logpdf(initial_sample)
        self.logpdf = logpdf # function to evaluate logpdf
        self.accept_num = 0
        self.num_steps = 0

    def accept_ratio(self):
        return self.accept_num / self.num_steps
    
    def update_state(self, new_sample_logpdf):
        accept = new_sample_logpdf[2]
        if accept is True:
            self.current_sample = new_sample_logpdf[0]
            self.current_logpdf = new_sample_logpdf[1]
            self.accept_num += 1
        self.num_steps += 1
        self.process_new_sample(self.current_sample, self.current_logpdf)

    @abc.abstractmethod
    def process_new_sample(self, sample, logpdf):
        """Do stuff with a sample, other than saving it (mainly for adaptation)
        """
        raise NotImplementedError("Should implement process_new_sample")
    
    @abc.abstractmethod
    def propose(self):
        """Propose a sample
        """
        raise NotImplementedError("Should implement step")

    def step(self):
        raise NotImplementedError("Need to implement non-symmetric proposal")
    
class MetropolisHastingsSym(MetropolisHastings, abc.ABC):
    """Symmetric Metropolis Hastings.
    
    This is an abstract class that assumes a symmetic proposal is used
    """
    
    def __init__(self, logpdf, initial_sample):
        super().__init__(logpdf, initial_sample)

    def step(self):
        proposed_sample = self.propose()
        prop_pdf = self.logpdf(proposed_sample)

        accept_reject = prop_pdf - self.current_logpdf
        if accept_reject > 0:
            return (proposed_sample, prop_pdf, True)
        else:
            u = np.log(np.random.rand(1)[0])
            if (u < accept_reject):
                return (proposed_sample, prop_pdf, True)
            else:
                return (self.current_sample, self.current_logpdf, False)

class RandomWalkGauss(MetropolisHastingsSym):
    """Random Walk metropolis Hastings Sampler."""
    
    def __init__(self, logpdf, initial_sample, cov):
        super().__init__(logpdf, initial_sample)
        self.dim = cov.shape[0]
        self.cov = cov
        self.cov_chol = np.linalg.cholesky(cov)

    def propose(self):
        return sample_gauss(self.current_sample, self.cov_chol)

    def process_new_sample(self, sample, logpdf_val):
        pass
    
class DelayedRejectionGauss(RandomWalkGauss):

    def __init__(self, logpdf, initial_sample, initial_cov, level_scale=1e-1):
        RandomWalkGauss.__init__(self, logpdf, initial_sample, initial_cov)
        self.gamma = level_scale
                
    def propose(self, level):
        if level == 0:
            return super().propose()
        else:
            return sample_gauss(self.current_sample, self.gamma * self.cov_chol)

    def step(self):
        proposed_sample = self.propose(0)
        prop_pdf = self.logpdf(proposed_sample)

        accept_reject = prop_pdf - self.current_logpdf
        if accept_reject > 0:
            return (proposed_sample, prop_pdf, True)
        else:
            u = np.log(np.random.rand(1)[0])
            if (u < accept_reject):
                return (proposed_sample, prop_pdf, True)
            else:
                second_proposed_sample = self.propose(1)
                second_proposed_pdf = self.logpdf(second_proposed_sample)

                a2 = min(1, np.exp(prop_pdf - second_proposed_pdf))
                if a2 > 1.0-1e-15: # reject
                    return (self.current_sample, self.current_logpdf, False)
                
                diff2 = second_proposed_sample - proposed_sample
                # change to use cov_chol
                gauss_pdf_num = -0.5 * np.dot(diff2, np.linalg.solve(self.cov, diff2))

                diff1 = self.current_sample - proposed_sample
                gauss_pdf_den = -0.5 * np.dot(diff1, np.linalg.solve(self.cov, diff1))

                # print("a2 = ", a2)
                a2 = accept_reject + gauss_pdf_num - gauss_pdf_den + np.log(1.0 - a2) - \
                    np.log(1 - min(1, np.exp(accept_reject)))

                if a2 > 0:
                    return (second_proposed_sample, second_proposed_pdf, True)
                else:
                    u = np.log(np.random.rand(1)[0])
                    if (u < accept_reject):
                        return (second_proposed_sample, second_proposed_pdf, True)
                    else:
                        return (self.current_sample, self.current_logpdf, False)


class AdaptiveMetropolisGauss(RandomWalkGauss):
    """Adaptive Metropolis with Gaussian proposal sampler."""

    def init_adaptive(self, adapt_start, initial_sample, eps, sd, interval):
        """Initialize parameters for adaptive metropolis."""
        self.k = 0
        if adapt_start is not None:
            self.adapt_start = adapt_start
        else:
            self.adapt_start = len(initial_sample)

        self.mean = copy.deepcopy(initial_sample)
        self.epsI = eps * np.eye(self.dim)
        if sd is not None:
            self.sd = sd
        else:
            self.sd = 2.4**2 / self.dim
        self.S = np.zeros(self.dim)
        self.interval = interval

    def __init__(self, logpdf, initial_sample, initial_cov, adapt_start=None,
                 eps=1e-8, sd=None, interval=1):
        """Initialize."""
        RandomWalkGauss.__init__(self, logpdf, initial_sample, initial_cov)
        self.init_adaptive(adapt_start, initial_sample, eps, sd, interval)

    def compute_update_mean(self, new_pt):
        """Recursively compute the updated mean."""
        new_mean = (self.mean * self.k + new_pt) / (self.k+1)
        return new_mean

    def update_cov_and_mean(self, new_pt):
        """Recursively update hte covariance and mean."""
        next_mean = self.compute_update_mean(new_pt)

        t1 = self.k * np.outer(self.mean, self.mean) - \
            (self.k + 1) * np.outer(next_mean, next_mean) + \
            np.outer(new_pt, new_pt) + self.epsI 
        t1 *= self.sd / self.k 
        self.S = (self.k - 1)/self.k * self.S + t1
        self.mean = next_mean

    def process_new_sample(self, sample, logpdf_val):
        """Update internal data with new sample."""
        self.k += 1
        self.update_cov_and_mean(sample)

        # if we are beyond adapt_start, then use new covariance
        if self.k > self.adapt_start and self.k % self.interval == 0:
            # print("update!")
            self.cov = self.S
            self.cov_chol = np.linalg.cholesky(self.S)


class DelayedRejectionAdaptiveMetropolis(DelayedRejectionGauss,
                                         AdaptiveMetropolisGauss):
    """Delayed Rejection Adaptive Metropolis."""

    def __init__(self, logpdf, initial_sample, initial_cov, adapt_start=None,
                 eps=1e-8, sd=None, interval=1, level_scale=1e-1):
        """Initialize."""
        DelayedRejectionGauss.__init__(self, logpdf, initial_sample,
                                       initial_cov, level_scale=level_scale)
        AdaptiveMetropolisGauss.init_adaptive(self, adapt_start,
                                              initial_sample,
                                              eps, sd, interval)

        
if __name__ == "__main__":
    # things below are just tests during development
    np.random.seed(5)
    import itertools
    import pandas as pd
    import scipy.stats as scistats
    import matplotlib.pyplot as plt
        
    print("Hello World!")
    mc = MonteCarloSampler(lambda : np.random.randn(1))

    samples = itertools.islice(mc, 10)
    for idx, sample in enumerate(samples):
        print(f"sample[{idx}] = {sample}")

    print("Metropolis Hastings")

    cov = np.array([[1.0, 0.3],
                    [0.3, 2.0]])
    chol = np.linalg.cholesky(cov)
    # print(np.dot(chol, chol.T))
    # exit(1)
    mean = np.array([0.2, 0.5])
    logpdf_f = lambda x: scistats.multivariate_normal.logpdf(x, mean=mean, cov=cov)
    num_samples = 50000
    # mh = RandomWalkGauss(logpdf_f, mean, 0.2*np.eye(2));
    adapt_start = 5
    # mh = AdaptiveMetropolisGauss(logpdf_f, mean, 0.2*np.eye(2), adapt_start,
    #                              eps=1e-6, sd=2.4**2 / cov.shape[0],
    #                              interval=10);

    # mh = DelayedRejectionGauss(logpdf_f, mean, 0.2*np.eye(2), 1e-1)
    mh = DelayedRejectionAdaptiveMetropolis(logpdf_f, mean, 0.2*np.eye(2), adapt_start=10,
                                            eps=1e-6, sd=2.4**2 / cov.shape[0],
                                            interval=10, level_scale=1e-1)
    # exit(1)
    # for sample in mh:
    #     print(sample)

    mh2 = itertools.starmap(lambda x,y,z: x, itertools.islice(mh, num_samples))

    df = pd.DataFrame(mh2)
    arr = df.to_numpy()
    print(arr.shape)
    print("cov = \n", df.cov().to_numpy())
    print("mean = ", df.mean().to_numpy())
    # print(np.cov(arr, rowvar=False))
    # print("mh cov = ", mh.cov)

