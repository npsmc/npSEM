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
# The npSEM algorithm is run to reconstruct the model $m(x) = sin(3x)$ and estimate the parameter $\theta = (Q, R) \in \mathbb{R} \times \mathbb{R}$ given the observed sequence. Its results are compared to the ones derived from a SEM algorithm, SEM($m$), where both the model $m$ and the observations are provided, and the ones of another SEM algorithm, SEM($\hat m$) where both an estimate of $m$ learned on a sequence of the state process  and the observations are provided.

# In[1]:


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
from npsem.methods import _CPF_BS
from npsem.methods import CPF_BS_SEM
from npsem.methods import LLR_CPF_BS_SEM
from npsem.methods import _EnKS
from npsem.methods import RMSE

from npsem import TimeSeries
from npsem.models import SSM


# In[2]:

### GENERATE SIMULATED DATA (SINUS MODEL)

# parameters
dx = 1 # dimension of the state
dt_int = 1 # fixed integration time
dt_model = 1 # chosen number of model time step, the larger dt_model the more nonliner model
var_obs = np.array([0]) # indices of the observed variables
dy = len(var_obs) # dimension of the observations
H = np.eye(dx)
h = lambda x: H.dot(x)  # observation model

a = 3
mx = lambda x: np.sin(a*x) 
jac_mx = lambda x: a * np.cos(a*x) 

# Setting covariances
sig2_Q = 0.1; sig2_R = 0.1 # parameters
Q_true = np.eye(dx) *sig2_Q #  model variance
R_true = np.eye(dx) *sig2_R # observation variance

def generate_data(x0,f,h,Q,R,dt_int,dt_model,var_obs, T_burnin, T_train, T_test):
    """ Generate the true state, noisy observations and catalog of numerical simulations. """

    # initialization
    class X_train:
        values=[]
        time=[];
    class Y_train:
        values=[]
        time=[];
    class X_test:
        values = [];
        time = [];
    class Y_test:
        values = [];
        time = [];
    
    
#    # test on parameters
#    if dt_model>dt_obs:
#        print('Error: dt_obs must be bigger than dt_model');
#    if (np.mod(dt_obs,dt_model)!=0):
#        print('Error: dt_obs must be a multiple of dt_model');

    np.random.seed(1)
    # 5 time steps (to be in the attractor space)
    dx = x0.size
    x = np.zeros((dx,T_burnin))
    x[:,0] = x0
    for t in range(T_burnin-1):
        xx = x[:,t]
        for  i in range(dt_model):
            xx = f(xx)
        x[:,t+1] = xx + np.random.multivariate_normal(np.zeros(dx),Q)
    x0 = x[:,-1];

    # generate true state (X_train+X_test)
    T = T_train+T_test
    X = np.zeros((dx,T))
    X[:,0] = x0
    for t in range(T-1):
        XX = X[:,t]
        for  i in range(dt_model):
            XX = f(XX)
        X[:,t+1] = XX + np.random.multivariate_normal(np.zeros(dx),Q)
    # generate  partial/noisy observations (Y_train+Y_test)    
    Y = X*np.nan
    yo = np.zeros((dx,T))
    for t in range(T-1):
        yo[:,t]= h(X[:,t+1]) + np.random.multivariate_normal(np.zeros(dx),R)
    Y[var_obs,:] = yo[var_obs,:];
    
    # Create training data (catalogs)
    ## True catalog
    X_train.time = np.arange(0,T_train*dt_model*dt_int,dt_model*dt_int);
    X_train.values = X[:, 0:T_train];
    ## Noisy catalog
    Y_train.time = X_train.time[1:];
    Y_train.values = Y[:, 0:T_train-1]

    # Create testinging data 
    ## True catalog
    X_test.time = np.arange(0,T_test*dt_model*dt_int,dt_model*dt_int);
    X_test.values = X[:, T-T_test:]; 
    ## Noisy catalog
    Y_test.time = X_test.time[1:];
    Y_test.values = Y[:, T-T_test:-1]; 
 
    

    # reinitialize random generator number
    np.random.seed()

    return X_train, Y_train, X_test, Y_test,yo;


# prior state
x0 = np.ones(1)

# generate data
T_burnin = 10**3
T_train = 1000# length of the training
T_test = 10**3 # length of the testing data
X_train, Y_train, X_test, Y_test, yo = generate_data(x0,mx,h,Q_true,R_true,dt_int,dt_model,var_obs, T_burnin, T_train, T_test) 
X_train.time = np.arange(0,T_train)
Y_train.time= X_train.time[1:]
X_test.time = np.arange(0,T_test)
Y_test.time= X_test.time[1:]

X_train0, Y_train0, X_test0, Y_test0, yo0 = generate_data(X_test.values[:,-1],mx,h,Q_true,R_true,dt_int,dt_model,var_obs, T_burnin, T_train, T_test) 



### PLOT STATE, OBSERVATIONS AND CATALOG

# state and observations (when available)
sns.set_style("white")
plt.rcParams['figure.figsize'] = (15, 4)
plt.figure(1)
plt.plot(X_train.values[:,1:].T,'-', color='b')
plt.plot(Y_train.values.T,'.b', markersize= 6)
plt.legend(['state','observations'], ncol=2)
plt.xlabel('time(t)')
plt.ylabel('space')
plt.title('Time seies of the state and observed trajectories of the sinus model')
plt.xlim([0,100])
plt.grid()
plt.show()


