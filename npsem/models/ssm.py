import sys, os

import matplotlib.pyplot as plt
import numpy as np
from numpy.random import multivariate_normal

import .l63f as mdl_l63
from .l63 import l63_jac
from ..time_series import TimeSeries

class SSM:
    """
    Space State Model
     
    :param dx      : dimension of the state
    :param var_obs : indices of the observed variables
    :param ny      : dimension of the observations
    :param Q       : model covariance
    :param R       : observation covariance
    """

    def __init__(self, dx, var_obs, sig2_Q = 1, sig2_R = 1 ):

        self.dx       = dx
        self.var_obs  = var_obs
        self.dy       = len(var_obs)
        self.H        = np.eye(dx)
        self.sig2_Q   = sig2_Q
        self.sig2_R   = sig2_R  
        self.Q        = np.eye(dx) * sig2_Q
        self.R        = np.eye(dx) * sig2_R

def plot(X, Y, figsize=(15,5)):

    fig, axes = plt.subplots(figsize = (15, 5))
    axes.plot(X.values[:,1:].T,'-', color='grey')
    axes.plot(Y.values.T,'.k', markersize= 6)
    axes.set_xlabel('Lorenz-63 times')
    axes.set_title('Lorenz-63 true (continuous lines) '
                   'and observed trajectories (points)')
    return axes

def train_test_split( X, Y, test_size=0.5):

    assert isinstance(TimeSeries, X)
    assert isinstance(TimeSeries, Y)

    time = X.time
    T    = time.size
    T_test  = int(T * test_size)
    T_train = T - T_test

    X_train = TimeSeries( time[0:T_train], X.values[:, 0:T_train])
    Y_train = TimeSeries( time[1:], Y.values[:, 0:T_train - 1])

    X_test = TimeSeries(time,     X.values[:, T - T_test:])
    Y_test = TimeSeries(time[1:], Y.values[:, T - T_test:-1])

    return X_train, Y_train, X_test, Y_test

if __name__ == "__main__":

    generate_data = Lorenz63()
    T_burnin = 5  * 10**3
    T        = 20 * 10**2
    X, Y = generate_data( T_burnin, T)

    X_train, Y_train, X_test, Y_test = train_test_split(X, Y, test_size=0.5)

    plot_data(X_train, Y_train)
    plt.show()

