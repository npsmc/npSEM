#!/usr/bin/env python

""" AnDA_stat_functions.py: Collection of statistical functions used in AnDA. """

__author__ = "Pierre Tandeo and Phi Huynh Viet"
__version__ = "1.0"
__date__ = "2016-10-16"
__maintainer__ = "Pierre Tandeo"
__email__ = "pierre.tandeo@telecom-bretagne.eu"

import numpy as np
import sys
from numba import jit

def CP95(Xs,X):
    dx, T = X.shape
    cov_prob = np.zeros((dx,1))
    CIlowXs= np.percentile(Xs, 2.5, axis= 1)
    CIupXs = np.percentile(Xs, 97.5, axis= 1)
    for var in range(dx):
        cov_prob1= np.array(np.where(CIlowXs[var,:] < X[var,:]))
        cov_prob2= np.array(np.where(X[var,:]< CIupXs[var,:]))
        cov_prob[var]= len(np.intersect1d(cov_prob1,cov_prob2))/T*100
    return cov_prob, CIlowXs, CIupXs

def RMSE(error):
  return np.sqrt(np.mean(error**2))

@jit
def sampling_discrete(W, m):  ### Discrete sampling, ADDED by TRANG ###
    "Returns m indices given N weights"
    cumprob = np.cumsum(W)
    n = W.size
    R = np.random.rand(m)
    ind = np.zeros(m, dtype = int)
    for i in range(n):
        ind += R > cumprob[i]
    return ind


def resampling_sys(W): ### systematic resampling with respect to multinomial distribution, ADDED by TRANG ###
    "Returns a N-set of indices given N weights"
    N = np.size(W)
    u0 = np.random.rand(1)
    u = (range(N) +u0).T/N;
    qc = np.cumsum(W)
    qc = qc[:]
    qc = qc/qc[-1]
    new= np.concatenate((u,qc), axis=0)
    ind1 = np.argsort(new)
#   ind2 = np.where(ind1<=N-1);
    ind2 = np.array(np.where(ind1<=N-1),dtype = int)
    ind = ind2- range(N)
    a = ind[0,]
    return a
