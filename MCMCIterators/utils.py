"""Utilities for SamplingIterators."""
import numpy as np
import scipy.optimize

def laplace_approx(initial_guess, logpost):
    """Perform the laplace approximation
    returning the MAP point and an approximation of the covariance

    Parameters
    ----------
    initial_guess : 1-D array (nparam , ) 
                   Array of initial parameters
    logpost : function (param) -> float
              log posterior

    Returns
    ------
    map_point : 1-D array (nparam , ) 
                MAP of the posterior
    cov_approx: 2-D array (nparam , nparam) 
                Covariance matrix for Gaussian fit at MAP

    Notes
    -----
    One must visually inspec thte output of this function, it is not always
    reliable and multiple runs may be needed to find a good MAP and covariance
    """
    def neg_post(x):
        """Negative posteror because optimizer is a minimizer."""
        return -logpost(x)

    # Gradient free method to obtain optimum
    res = scipy.optimize.minimize(neg_post, initial_guess, method='Nelder-Mead')
    # Gradient method which also approximates the inverse of the hessian
    res = scipy.optimize.minimize(neg_post, res.x)

    map_point = res.x
    cov_approx = res.hess_inv
    return map_point, cov_approx


def auto_correlation(samples):
    """Compute Autocorrelation.

    Parameters
    ----------
    samples : 1-D array
              Array of samples whose autocorrelation we seek

    Returns
    -------
    Array of autocorrelations
    """
    xp = (samples - np.mean(samples)) / np.std(samples)
    result = np.correlate(xp, xp, mode='full')
    return result[int(result.size/2):] / xp.shape[0]
