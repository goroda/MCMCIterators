# MCMC-Iterators: Markov Chain Monte Carlo Iterators
Iterator-based Markov Chain Monte Carlo Sampling Algorithms

## Why this package?
Packages like pymc3, pyro, and stan trap you within their modeling world. Sometimes you just want a quick and easy sampling approach that takes a minimal amount of overhead. This package is created so that all you need to provide is a user-defined distribution (log density) from which you would like to sample. This is just a standard python function and does not trap you into any specific code-base. The algorithms in this package allow you to then create a loop generating samples, just like a regular loop in python. You can do whatever you want with the samples. 

The currently implemented algorithms implemented for MCMC
* Random Walk Metropolis Hastings (MH)
* Adaptive Metropolis (AM)
* Delayed Rejection Metropolis Hastings  (DR)
* Delayed Rejection Adaptive Metropolis (DRAM) 
    
    
## Example

To show you the ease of use of this package, lets use DRAM to generate samples from a simple 2D gaussian.
First you construct a function to evaluate the logpdf of a shifted and skewed Gaussian

    def gauss_logpdf(x):
        cov = np.array([[1.0, 0.2], 
                         0.2, 2.0])
        mean = np.array([1.0, 2.0])
        
        diff = x - mean
        return -0.5 * np.dot(diff,  np.linalg.solve(cov, diff))
        
Next we setup the sampler. For the initial sample we will use a zero vector and for the initial covariance the identity. Though you can use whatever you want, and this package does include some code for computing a Laplace Approximation.


    init_sample = np.array([0.0, 0.0])
    init_cov = np.array([[1.0, 0.0], 
                         [0.0, 1.0]])
                         
Finally, we setup the DRAM sampler, here I expose all the parameters. In practice you can use the defaults.
    
    sampler = DelayedRejectionAdaptiveMetropolis(gauss_logpdf, init_sample, init_cov
                                                 adapt_start=10,
                                                 eps=1e-6, sd=None
                                                 interval=1, level_scale=1e-1)


This is now a simple iterator. So you can iterate forever with

    for sample, logpdf, accepted_bool in sampler:
        print(f"Sample: {sample}")
        print(f"\t Logpdf: {logpdf}")
        print(f"\t Accepted? -> {accepted_bool}")
        print("\n")
        
Notice that the sampler outputs three things: the next sample, the evaluation of the logpdf, and whether or not this is a new sample that was accepted (True) or an old sample wjere a new sample was proposed, but rejected (False)

One can use this iterator in conjuction with any itertools functions. For instance the following function turns this iterator into one with a finite number of samples (100)

    sampler = itertools.islice(sampler, 100)
    
One can then initialize a Panda data frame with an iterator.

    df = pd.DataFrame(samples, columns=['x1', 'x2'])
    
For more examples please checkout the [banana function](examples/banana.py)

## Installation

This package omes with both a library and some examples. The library must be installed prior to running the examples. To install the library, one can run the following commands

	python setup.py build
	python setup.py install

## Unit testing

To run unit tests one can run

    python -m unittest discover

from the top level directory.

## Information
author: Alex Gorodetsky  
email:  alex@alexgorodetsky.com  
license: GPL3  
copyright (c) 2022, Alex Gorodetsky  
