"""Collection of Markov Chain Monte Carlo Algorithms as Iterators."""
import abc
import numpy as np
import copy
from collections import namedtuple

# StopIteration # <-- To stop an iteration inside next

"""Result type"""
Sample = namedtuple("Sample", ['sample', 'logpdf', 'accepted_bool', 'info'])
SamplerState = namedtuple("SamplerState", ["cov", "proposed_samples", "proposed_logpdfs"])

def sample_gauss(mean, cov_chol):
    """Sample a gaussian with given mean and covariance square root."""
    return mean + np.dot(cov_chol, np.random.randn(mean.shape[0]))


class SequentialSampler(abc.ABC):
    """Core Base Class of a Sequential Sampler."""

    @abc.abstractmethod
    def update_state(self, new_sample):
        """Update the state with the new sample."""
        raise NotImplementedError("Should implement update_state")

    @abc.abstractmethod
    def step(self):
        """Produce the next sample."""
        raise NotImplementedError("Should implement step")

    def __iter__(self):
        """Make this an iterator."""
        return self

    def __next__(self):
        """Take another step, returning a new sample."""
        new_step = self.step()
        self.update_state(new_step)
        return new_step


class MonteCarloSampler(SequentialSampler):
    """
    Monte Carlo is implemented as SequentialSampler.

    Obviously you would not want to do this,
    because Monte Carlo can be vectorized and done in parallel.
    However, this can be convenient for some things.
    """

    def __init__(self, sampler):
        """Initialize with a sampler."""
        self.sampler = sampler

    def update_state(self, new_sample):
        """Don't update any state."""
        pass

    def step(self):
        """Call the sampler and return the result."""
        return self.sampler()


class MetropolisHastings(SequentialSampler, abc.ABC):
    """
    Metropolis Hastings Sampler.

    This is an abstract class that specifies the required
    functions that must be filled in for a concrete realization

    In particular, the concrete classes will need to implement the
    following methods

    process_new_sample
    propose
    step
    """

    def __init__(self, logpdf, initial_sample):
        """Initialize.

        Parameters
        ----------
        logpdf : callable, 1-D array -> float
                 Density from which to sample

        initial_sample : 1-D array
                         initial sample
        """
        self.current_sample = initial_sample
        self.current_logpdf = logpdf(initial_sample)
        self.logpdf = logpdf  # function to evaluate logpdf
        self.accept_num = 0
        self.num_steps = 0

    def accept_ratio(self):
        """Compute Acceptance Ratio."""
        return self.accept_num / self.num_steps

    def update_state(self, new_sample_logpdf):
        """Update the state.

        Store the sample, update the acceptance ratio,
        and process the new sample

        Parameters
        ----------
        new_sample_logpdf : tuple (sample, logpdf_value)
        """
        accept = new_sample_logpdf[2]
        if accept is True:
            self.current_sample = new_sample_logpdf[0]
            self.current_logpdf = new_sample_logpdf[1]
            self.accept_num += 1
        self.num_steps += 1
        self.process_new_sample(self.current_sample, self.current_logpdf)

    @abc.abstractmethod
    def process_new_sample(self, sample, logpdf):
        """Do stuff with a sample, e.g., update some state."""
        raise NotImplementedError("Should implement process_new_sample")

    @abc.abstractmethod
    def propose(self):
        """Propose a sample."""
        raise NotImplementedError("Should implement propose")

    def step(self):
        """Take a step."""
        raise NotImplementedError("Should implement step")

class HMC(MetropolisHastings):
    """
    Hamiltonian Monte Carlo Sampler
    """
    def __init__(self, logpdf, initial_sample, grad_logpdf, step_size, num_steps):
        
        super().__init__(logpdf, initial_sample)
        self.grad_logpdf = grad_logpdf
        self.step_size = step_size
        self.num_steps = num_steps

    def leapfrog(self, q, p, step):
        pn = p + 0.5 * step * self.grad_logpdf(q)
        qn = q + step * pn
        pn = pn + 0.5 * step * self.grad_logpdf(qn)
        return qn, pn
    
    def hamiltonian(self, q, p):
        return self.logpdf(q) - 0.5 * np.dot(p, p)

    def propose(self):

        q = np.copy(self.current_sample)        
        r0 = np.random.randn(self.dim)
        r = np.copy(r0)

        for ii in range(self.num_steps):
            q, r = self.leapfrog(q, r, self.step_size)

        return q, r, r0

    def process_new_sample(self, sample, logpdf):
        pass
    
    def step(self, qin):

        q, r, r0 = self.propose()
        H0 = self.current_logpdf - 0.5 * np.dot(r0, r0)
        prop_logpdf = self.logpdf(q)
        H1 = prop_logpdf - 0.5 * np.dot(r, r)

        u = np.log(np.random.rand(1)[0])

        state = SamplerState(proposed_samples=[q], proposed_logpdfs=[prop_logpdf], cov = None)
        if u < H1 - H0:
            return Sample(q, prop_logpdf, True, state)
        else:
            return Sample(self.current_sample, self.current_logpdf, False, state)

    
class MetropolisHastingsSym(MetropolisHastings, abc.ABC):
    """
    Symmetric Metropolis Hastings.

    This is an abstract class that assumes a symmetic proposal is used.
    """

    def __init__(self, logpdf, initial_sample):
        """Initialize with same args as parent class MetropolisHastings."""
        super().__init__(logpdf, initial_sample)

    def step(self):
        """Propose a sample and accept/reject it.

        Returns
        -------
        Triple of (next sample, sample logpdf, accepted?(bool))
        """
        proposed_sample = self.propose()
        prop_pdf = self.logpdf(proposed_sample)

        accept_reject = prop_pdf - self.current_logpdf
        state = SamplerState(proposed_samples=[proposed_sample], proposed_logpdfs=[prop_pdf], cov = None)
        if accept_reject > 0:
            return Sample(proposed_sample, prop_pdf, True, state)
        else:
            u = np.log(np.random.rand(1)[0])
            if u < accept_reject:
                return Sample(proposed_sample, prop_pdf, True, state)
            else:
                return Sample(self.current_sample, self.current_logpdf, False, state)


class RandomWalkGauss(MetropolisHastingsSym):
    """Random Walk metropolis Hastings Sampler."""

    def __init__(self, logpdf, initial_sample, cov):
        """Initialize a random walk sampler with proposal covariance *cov*."""
        super().__init__(logpdf, initial_sample)
        self.dim = cov.shape[0]
        self.cov = cov
        self.cov_chol = np.linalg.cholesky(cov)

    def propose(self):
        """Generate proposed sample."""
        return sample_gauss(self.current_sample, self.cov_chol)

    def process_new_sample(self, sample, logpdf):
        """Don't process the sample."""
        pass

class DelayedRejectionGauss(RandomWalkGauss):
    """Delayed Rejection Metropolis Hastings."""

    def __init__(self, logpdf, initial_sample, initial_cov, level_scale=1e-1):
        """
        Initialize with same arguments as RandomWalkGauss.

        Covariance scaling set to *level_scale* for second level proposal
        """
        RandomWalkGauss.__init__(self, logpdf, initial_sample, initial_cov)
        self.gamma = level_scale

    def propose(self, level):
        """Propose a new sample from a given level."""
        if level == 0:
            return super().propose()
        else:
            return sample_gauss(self.current_sample,
                                np.sqrt(self.gamma) * self.cov_chol)

    def step(self):
        """Generate the next sample."""
        proposed_sample = self.propose(0)
        prop_pdf = self.logpdf(proposed_sample)

        accept_reject = prop_pdf - self.current_logpdf
        state = SamplerState(proposed_samples=[proposed_sample], proposed_logpdfs=[prop_pdf], cov = self.cov)
        if accept_reject > 0:
            return Sample(proposed_sample, prop_pdf, True, state)
        else:
            u = np.log(np.random.rand(1)[0])
            if u < accept_reject:
                return Sample(proposed_sample, prop_pdf, True, state)
            else:
                second_proposed_sample = self.propose(1)
                second_proposed_pdf = self.logpdf(second_proposed_sample)
                state.proposed_samples.append(second_proposed_sample)
                state.proposed_logpdfs.append(second_proposed_pdf)

                a2 = min(1, np.exp(prop_pdf - second_proposed_pdf))
                if a2 > 1.0-1e-15:  # reject
                    return Sample(self.current_sample, self.current_logpdf, False, state)

                diff2 = second_proposed_sample - proposed_sample
                # change to use cov_chol
                gauss_pdf_num = -0.5 * np.dot(diff2,
                                              np.linalg.solve(self.cov, diff2))

                diff1 = self.current_sample - proposed_sample
                gauss_pdf_den = -0.5 * np.dot(diff1,
                                              np.linalg.solve(self.cov, diff1))

                # print("a2 = ", a2)
                a2 = second_proposed_pdf - self.current_logpdf + \
                    gauss_pdf_num - gauss_pdf_den + \
                    np.log(1.0 - a2) - \
                    np.log(1 - min(1, np.exp(accept_reject)))

                if a2 > 0:
                    return Sample(second_proposed_sample, second_proposed_pdf, True, state)
                else:
                    u = np.log(np.random.rand(1)[0])
                    if (u < accept_reject):
                        return Sample(second_proposed_sample, second_proposed_pdf, True, state)
                    else:
                        return Sample(self.current_sample, self.current_logpdf, False, state)


class AdaptiveMetropolisGauss(RandomWalkGauss):
    """Adaptive Metropolis with Gaussian proposal sampler."""

    def init_adaptive(self, adapt_start, initial_sample, eps, sd, interval):
        """Initialize parameters for adaptive metropolis.

        Parameters
        ----------
        adapt_start : int
                      Number of iterations after which to start using
                      adapted covariance

        initial_sample : 1-D array
                         Initial sample

        eps : float
              Small number for covariance scaling nugget

        sd : float
             Sample covariance scaling

        interval : int
             Interval of adaptation
        """
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
        self.S = np.zeros((self.dim, self.dim))
        self.interval = interval

    def __init__(self, logpdf, initial_sample, initial_cov, adapt_start=None,
                 eps=1e-8, sd=None, interval=1):
        """Initialize. See *init_adaptive* for input description"""
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

    mc = MonteCarloSampler(lambda: np.random.randn(1))
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
    mh = DelayedRejectionAdaptiveMetropolis(logpdf_f, mean, 0.2*np.eye(2),
                                            adapt_start=10,
                                            eps=1e-6, sd=2.4**2 / cov.shape[0],
                                            interval=10, level_scale=1e-1)
    # exit(1)
    # for sample in mh:
    #     print(sample)

    mh2 = itertools.starmap(lambda x, y, z, s: x, itertools.islice(mh, num_samples))

    df = pd.DataFrame(mh2)
    arr = df.to_numpy()
    print(arr.shape)
    print("cov = \n", df.cov().to_numpy())
    print("mean = ", df.mean().to_numpy())
    # print(np.cov(arr, rowvar=False))
    # print("mh cov = ", mh.cov)
