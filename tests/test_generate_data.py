#!/usr/bin/env python
# coding: utf-8
# %%
import numpy as np 
import matplotlib.pyplot as plt

import os, sys
cwd = os.getcwd()
sys.path.insert(0,os.path.join(cwd))

import noisette
from noisette.models import Lorenz63

dt_model = 8           # chosen number of model time step  \in [1, 25]
                       # the larger dt_model the more nonlinear model
var_obs = np.arange(3) # indices of the observed variables
sigma   = 10.0
rho     = 28.0
beta    = 8/3          # physical parameters

# Setting covariances
sig2_Q = 1
sig2_R = 2 # parameters

# prior state
model = Lorenz63(x0       = [8, 0, 30],
                 dt_int   = 0.01, 
                 dt_model = 8,    
                 var_obs  = np.arange(3), 
                 sigma    = 10.0,
                 rho      = 28.0,
                 beta     = 8/3,          
                 sig2_Q   = sig2_Q,
                 sig2_R   = sig2_R )

# generate data
T_burnin = 5*10**3
T        = 20*10**2 # length of the catalog

X, Y, = model( T_burnin, T) 

plt.rcParams['figure.figsize'] = (15, 5)
plt.figure(1)
plt.plot(X.values[:,1:].T,'-', color='grey')
plt.plot(Y.values.T,'.k', markersize= 6)
plt.xlabel('Lorenz-63 times')
plt.title('Lorenz-63 true (continuous lines) and observed trajectories (points)')
plt.show()
