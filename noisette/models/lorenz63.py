import sys, os

import matplotlib.pyplot as plt
import numpy as np
from numpy.random import multivariate_normal

import noisette.models.l63f as mdl_l63
from noisette.models.l63 import l63_jac
from noisette.time_series import TimeSeries

class Lorenz63:
    """
     GENERATE SIMULATED DATA (LORENZ-63 MODEL)
     
    :param dx                 : dimension of the state
    :param dt_int             : fixed integration time
    :param dt_model           : chosen number of model time step  \in [1, 25]
                                the larger dt_model the more nonlinear model
    :param var_obs            : indices of the observed variables
    :param dy                 : dimension of the observations
    :param H                  : first and third variables are observed
    :param h                  : observation model
    :param jacH               : Jacobian of the observation model(for EKS_EM only)
    :param sigma, rho, beta   : physical parameters
    :param Q                  : model covariance
    :param R                  : observation covariance
    """

    def __init__(self, dt_int   = 0.01,
                       dt_model = 8,
                       var_obs  = [0, 1, 2],
                       sigma    = 10.0,
                       rho      = 28.0,
                       beta     = 8.0 / 3,
                       sig2_Q   = 1,
                       sig2_R   = 2, 
                       x0       = [8, 0, 30]):

        self.x0       = np.array(x0)
        dx            = len(x0)
        self.dt_int   = dt_int
        self.dt_model = dt_model
        self.var_obs  = np.array(var_obs)
        self.H        = np.eye(dx)  # H = H[(0,2),:]
        self.dy       = self.var_obs.size
        self.sigma    = sigma
        self.rho      = rho
        self.beta     = beta
        # Setting covariances
        self.sig2_Q   = sig2_Q
        self.sig2_R   = sig2_R  
        self.Q        = np.eye(dx) * sig2_Q
        self.R        = np.eye(dx) * sig2_R
        self.fmdl     = mdl_l63.M(sigma, rho, beta, dt_int)

    def h(self, x):
        return self.H.dot(x)

    def jacH(self, x):
        return self.H

    def mx(self, x): 
        " fortran version (fast) "
        return self.fmdl.integ(x)  

    def jac_mx(self, x): 
        " python version (slow) "
        return l63_jac(x, self.dt_int * self.dt_model, 
                       self.sigma, self.rho, self.beta)  

    def __call__(self, T_burnin, T, seed = 1):
        """ 
        Generate simulated data (LORENZ-63 model)
        Generate the true state, noisy observations and catalog
        of numerical simulations.
        """

        np.random.seed(seed)

        dx = self.x0.size
        x = np.zeros((dx, T_burnin))
        x[:, 0] = self.x0
        for t in range(T_burnin - 1):
            xx = x[:, t]
            for i in range(self.dt_model):
                xx = self.mx(xx)
            x[:, t + 1] = xx + multivariate_normal(np.zeros(dx), self.Q)
        x0 = x[:, -1]

        # generate true state (X_train+X_test)
        X = np.zeros((dx, T))
        X[:, 0] = x0
        for t in range(T - 1):
            XX = X[:, t]
            for i in range(self.dt_model):
                XX = self.mx(XX)
            X[:, t + 1] = XX + multivariate_normal(np.zeros(dx), self.Q)

        # generate  partial/noisy observations (Y_train+Y_test)    
        Y = X * np.nan
        yo = np.zeros((dx, T))

        for t in range(T - 1):
            yo[:, t] = self.h(X[:, t + 1]) + multivariate_normal(np.zeros(dx), 
                                                                 self.R)

        Y[self.var_obs, :] = yo[self.var_obs, :]
        dt = self.dt_model * self.dt_int 
        time = np.arange(0, T * dt, dt)

        return TimeSeries(time, X), TimeSeries(time[1:], Y[:,:-1])
    
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

