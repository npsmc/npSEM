#!/usr/bin/env python
# coding: utf-8
# %%
import numpy as np 
import matplotlib.pyplot as plt


# %%
from noisette.methods.generate_data import generate_data

def test_generate_data():
    """ 
    GENERATE SIMULATED DATA (LORENZ-63 MODEL)
    """
    
    dx       = 3           # dimension of the state
    dt_int   = 0.01        # fixed integration time
    dt_model = 8           # chosen number of model time step  \in [1, 25]
                           # the larger dt_model the more nonlinear model
    var_obs = np.arange(3) # indices of the observed variables
    dy      = len(var_obs) # dimension of the observations
    H       = np.eye(dx)
    h       = lambda x: H.dot(x)  # observation model
    jacH    = lambda x: H         # Jacobian matrix  of the observation 
                                  # model(for EKS_EM only)
    sigma   = 10.0
    rho     = 28.0
    beta    = 8/3          # physical parameters
    fmdl    = mdl_l63.M(sigma=sigma, rho=rho, beta=beta, dtcy= dt_int)
    mx      = lambda x: fmdl.integ(x) # fortran
    jac_mx  = lambda x: l63_jac(x, dt_int*dt_model, sigma, rho, beta) # python
    
    # Setting covariances
    sig2_Q = 1
    sig2_R = 2 # parameters
    Q_true = np.eye(dx) * sig2_Q # model covariance
    R_true = np.eye(dx) * sig2_R # observation covariance
    
    # prior state
    x0 = np.r_[8, 0, 30]
    
    # generate data
    T_burnin = 5*10**3
    T_train  = 10*10**2 # length of the catalog
    T_test   = 10*10**2 # length of the testing data
    Xtrain, Ytrain, Xtest, Ytest, yo = generate_data(x0, mx, h, Q_true,
        R_true, dt_int, dt_model, var_obs, T_burnin, T_train, T_test) 
    Xtrain.time = np.arange(0,T_train)
    Ytrain.time= Xtrain.time[1:]
    N     = np.size(Ytrain.values)
    Ngap  = N//10 # create gaps: 10 percent of missing values
    indX  = np.random.choice(np.arange(N), Ngap, replace=False)
    ind_gap_taken = divmod(indX ,len(Ytrain.time))

    Ytrain.values[ind_gap_taken] = np.nan
    

    return Xtrain, Ytrain
    



# %%
Xtrain, Ytrain = test_generate_data()
plt.rcParams['figure.figsize'] = (15, 5)
plt.figure(1)
plt.plot(Xtrain.values[:,1:].T,'-', color='grey')
plt.plot(Ytrain.values.T,'.k', markersize= 6)
plt.xlabel('Lorenz-63 times')
plt.title('Lorenz-63 true (continuous lines) and observed trajectories (points)')

# %%
