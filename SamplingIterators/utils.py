import numpy as np
import scipy.optimize

def laplace_approx(initial_guess , logpost):
    """
    Perform the laplace approximation ,
    returning the MAP point and an approximation of the covariance

    Inputs
    ------
    initial_guess: (nparam , ) array of initial parameters
    logpost: function (param) -> log posterior

    Outputs
    ------
    map_point: (nparam , ) MAP of the posterior
    cov_approx: (nparam , nparam), covariance matrix for Gaussian fit at MAP
    """
    def neg_post(x):
        """ Negative posteror because optimizer is a minimizer """
        return -logpost(x)

    # Gradient free method to obtain optimum
    res = scipy.optimize.minimize(neg_post, initial_guess , method='Nelder-Mead')
    # Gradient method which also approximates the inverse of the hessian
    res = scipy.optimize.minimize(neg_post , res.x)

    map_point = res.x
    cov_approx = res.hess_inv
    return map_point , cov_approx

def auto_correlation(samples, column):
    """Compute Autocorrelation"""
    xp = (samples[:, column] - np.mean(samples[:, column])) / np.std(samples[:, column])
    result = np.correlate(xp, xp, mode='full')
    return result[result.size/2:] / xp.shape[0]

