import numpy as np
from numpy.random import multivariate_normal

from ..time_series import TimeSeries

class SSM:

    """
    Generate simulated data from Space State Model
     
    :param var_obs            : indices of the observed variables
    :param dy                 : dimension of the observations
    :param Q                  : model covariance
    :param R                  : observation covariance
    :param dx                 : dimension of the state
    :param dt_int             : fixed integration time
    :param dt_model           : chosen number of model time step  \in [1, 25]
                                the larger dt_model the more nonlinear model
    :param var_obs            : indices of the observed variables
    :param dy                 : dimension of the observations
    :param H                  : first and third variables are observed
    :param h                  : observation model
    :param jacH               : Jacobian of the observation model(for EKS_EM only)
    :param Q                  : model covariance
    :param R                  : observation covariance
    """

    def __init__(self, h, jac_h, mx, jac_mx,
                       dt_int   = 0.01,
                       dt_model = 8,
                       x0       = [8, 0, 30],
                       var_obs  = [0, 1, 2],
                       sig2_Q   = 1,
                       sig2_R   = 2):

        self.h        = h
        self.jac_h    = jac_h
        self.mx       = mx
        self.jac_mx   = jac_mx

        self.dt_int   = dt_int
        self.dt_model = dt_model
        self.dx       = len(x0)
        self.x0       = np.array(x0)
        self.dy       = len(var_obs)
        self.var_obs  = np.array(var_obs)
        self.sig2_Q   = sig2_Q
        self.sig2_R   = sig2_R  
        self.Q        = np.eye(self.dx) * sig2_Q
        self.R        = np.eye(self.dx) * sig2_R


    def generate_data(self, T_burnin, T, seed = 1):
        """ 
        Generate simulated data from Space State Model
        """

        np.random.seed(seed)

        x = np.zeros((self.dx, T_burnin))
        x[:, 0] = self.x0
        for t in range(T_burnin - 1):
            xx = x[:, t]
            for i in range(self.dt_model):
                xx = self.mx(xx)
            x[:, t + 1] = xx + multivariate_normal(self.dx*[0], self.Q)
        x0 = x[:, -1]

        # generate true state
        X = np.zeros((self.dx, T))
        X[:, 0] = x0
        for t in range(T - 1):
            XX = X[:, t]
            for i in range(self.dt_model):
                XX = self.mx(XX)
            X[:, t + 1] = XX + multivariate_normal(self.dx*[0], self.Q)

        # generate  partial/noisy observations
        Y = X * np.nan
        yo = np.zeros((self.dx, T))

        for t in range(T - 1):
            yo[:, t] = self.h(X[:, t + 1]) + multivariate_normal(self.dx*[0], 
                                                                 self.R)

        Y[self.var_obs, :] = yo[self.var_obs, :]
        dt = self.dt_model * self.dt_int 
        time = np.arange(0, T * dt, dt)

        return TimeSeries(time, X), TimeSeries(time[1:], Y[:,:-1])
    
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

