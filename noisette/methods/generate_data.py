#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 12:26:19 2018

@author: trang
"""

import numpy as np
from numpy.random import multivariate_normal
from noisette.time_series import TimeSeries

def generate_data(x0,
                  f,
                  h,
                  Q,
                  R,
                  dt_int,
                  dt_model,
                  var_obs,
                  T_burnin,
                  T_train,
                  T_test,
                  seed = 1):
    """
    Generate the true state, noisy observations and catalog
    of numerical simulations.
    """

    np.random.seed(seed)

    # 5 time steps (to be in the attractor space)
    dx = x0.size
    x = np.zeros((dx, T_burnin))
    x[:, 0] = x0
    for t in range(T_burnin - 1):
        xx = x[:, t]
        for i in range(dt_model):
            xx = f(xx)
        x[:, t + 1] = xx + multivariate_normal(np.zeros(dx), Q)
    x0 = x[:, -1]

    # generate true state (X_train+X_test)
    T = T_train + T_test
    X = np.zeros((dx, T))
    X[:, 0] = x0
    for t in range(T - 1):
        XX = X[:, t]
        for i in range(dt_model):
            XX = f(XX)
        X[:, t + 1] = XX + multivariate_normal(np.zeros(dx), Q)

    # generate  partial/noisy observations (Y_train+Y_test)    
    Y = X * np.nan
    yo = np.zeros((dx, T))

    for t in range(T - 1):
        yo[:, t] = h(X[:, t + 1]) + multivariate_normal(np.zeros(dx), R)

    Y[var_obs, :] = yo[var_obs, :]

    # Create training data (catalogs)

    # True catalog
    time = np.arange(0, T_train * dt_model * dt_int, dt_model * dt_int)
    X_train = TimeSeries(time, X[:, 0:T_train])
    # Noisy catalog
    Y_train = TimeSeries( time[1:], Y[:, 0:T_train - 1])

    # Create testinging data 

    # True catalog
    time = np.arange(0, T_test * dt_model * dt_int, dt_model * dt_int)
    X_test = TimeSeries(time, X[:, T - T_test:])
    # Noisy catalog
    Y_test = TimeSeries(time[1:], Y[:, T - T_test:-1])

    return X_train, Y_train, X_test, Y_test, yo
