#!/usr/bin/env python
# coding: utf-8

# ### Test SEM and npSEM algorithms on a 1d-sinus state-space model.
# Let us consider a sinus model
# \begin{equation} 
# \begin{cases}
# X_t =  sin(3X_{t-1}) +  \eta_t, \quad \eta_t \sim \mathcal{N} \left( 0, Q\right)\\
# Y_t =   X_t +  \epsilon _t, \quad   \epsilon _t \sim \mathcal{N} \left( 0, R\right).
# \end{cases}
# \end{equation}
# 
# Given true error variances  $(Q^*,R^*)=(0.1, 0.1)$,  sequences of the state process $(X_t)$ and the observations process $(Y_t)$ are simulated.
# 
# In[1]:


### IMPORT PACKAGES
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt 
import seaborn as sns

from npsem import TimeSeries
from npsem.models import SSM


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

# prior state
x0 = [1]

sinus3   = SSM(h, jac_h, mx, jac_mx, dt_int, dt_model, x0, var_obs, 
               sig2_Q, sig2_R)

# generate data
T_burnin = 10**3
T        = 2000# length of the training

X, Y     = sinus3.generate_data( T_burnin, T)


# state and observations (when available)
sns.set_style("white")
fig, axes = plt.subplots( 3, figsize = (15, 15))
axes[0].plot(X.time[1:], X.values[:,1:].T,'-', color='b')
axes[0].plot(Y.time, Y.values.T,'.b', markersize= 6)
plt.legend(['state','observations'], ncol=2)
axes[0].set_xlabel('time(t)')
axes[0].set_ylabel('space')
axes[0].set_title('Time series of the state and observed trajectories of the sinus model')
axes[0].set_xlim([0,100])
axes[1].scatter(X.values[:,:-1].ravel(), X.values[:,1:].ravel())
Xsorted = np.sort(X.values).ravel()
axes[1].plot(Xsorted, np.sin(3*Xsorted),'r-')
plt.grid()
axes[2].scatter(Y.values[:,:-1].ravel(), Y.values[:,1:].ravel())
plt.grid()
plt.savefig("sinus3_data.png")
