#!/usr/bin/env python
# coding: utf-8

### IMPORT PACKAGES
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt 
import seaborn as sns


#import routines
from npsem         import loadTr
from npsem.methods import generate_data
from npsem.methods import m_LLR
from npsem.methods import m_true
from npsem.methods import k_choice
from npsem.models  import SSM

from npsem import TimeSeries, Est


# In[2]:

### GENERATE SIMULATED DATA (SINUS MODEL)

# parameters
dx       = 1   # dimension of the state
dt_int   = 1   # fixed integration time
dt_model = 1   # chosen number of model time step
var_obs  = [0] # indices of the observed variables
dy       = len(var_obs) # dimension of the observations

H      = np.eye(dx)
h      = lambda x: H.dot(x)  # observation model
jac_h  = lambda x: x
a      = 3
mx     = lambda x: np.sin(a*x) 
jac_mx = lambda x: a * np.cos(a*x) 

# Setting covariances
sig2_Q = 0.1
sig2_R = 0.1 

Q = np.eye(dx) * sig2_Q # model covariance
R = np.eye(dx) * sig2_R # observation covariance


# prior state
x0 = [1]

sinus3   = SSM(h, jac_h, mx, jac_mx, dt_int, dt_model, x0, var_obs, 
               sig2_Q, sig2_R)

# generate data
T_burnin = 10**3
T        = 2000# length of the training

X, Y     = sinus3.generate_data( T_burnin, T)

X_train, Y_train, X_test, Y_test = train_test_split(X, Y, test_size=0.5)

# In[3]:

#%% FORECASTING
estQ  = Est( Q, 'fixed', 'constant', np.eye(dx), True)
estR  = Est([], 'fixed', 'constant', np.eye(dx), True) 
estX0 = Est(decision = False)
estD  = Est(decision = False)

num_ana_m = 200
num_ana_Q = 200

data_init = np.r_['0,2,0',Y_train.values[...,:-1], Y_train.values[...,1:]];


#  LLR_FORECASTING (non-parametric dynamical model constructed given the catalog)
# parameters of the analog forecasting method
class LLR:

    data = Data( dx, data_init, Y_train)

        ana = np.zeros((dx,1,len(ind_nogap)))
        suc = np.zeros((dx,1,len(ind_nogap)))
        ana[:,0,:] = data_init[:dx,ind_nogap]
        suc[:,0,:] = data_init[dx:,ind_nogap] 
        time = Y_train.time[ind_nogap] # catalog with analogs and successors

    class data_prev:
        ana  = data_init[:dx,ind_nogap]
        suc  = data_init[dx:,ind_nogap]
        time = Y_train.time[ind_nogap] # catalog with analogs and successors


    lag_x = 5 # lag of removed analogs around x to avoid an over-fitting forecast
    lag_Dx = lambda Dx: np.shape(Dx)[-1]; # length of lag of outline analogs removed
    time_period = 1 # set 365.25 for year and 
    k_m = [] # number of analogs 
    k_Q = [] # number of analogs 
    nN_m = np.arange(10,num_ana_m,20) #number of analogs to be chosen for mean estimation
    nN_Q = np.arange(10,num_ana_Q,20) #number of analogs to be chosen for dynamical error covariance
    lag_k = 1; #chosen the lag for k reestimation 
    k_lag = 10
    k_inc = 5
    estK = 'same' # set 'same' if k_m = k_Q chosen, otherwise, set 'different'
    kernel = 'tricube'# set 'rectangular' or 'tricube'
    Q = estQ;
    gam = 1;


# In[4]:


# SETTING PARAMETERS 
X_conditioning = np.zeros([dx,T_train+1]) # the conditioning trajectory 
B = Q_true
xb= X_train.values[...,0]
Nf = 10# number of particles
Ns = 5 # number of realizations

N_iter =500# number of iterations of EM algorithms
# Step functions
gam1 = np.ones(N_iter,dtype=int)  #for SEM (stochastic EM)
gam2 = np.ones(N_iter, dtype=int)
for k in range(50,N_iter):
    gam2[k] = k**(-0.7) # for SAEM (stochastic approximation EM)
#gam1 = gam2

# initial parameters
aintQ = 0.5; bintQ = 5
aintR = 1; bintR = 5
R_init = .5*np.eye(dy)    
Q_init = 3*np.eye(dx)
LLR.Q.value = Q_init


# In[5]:
