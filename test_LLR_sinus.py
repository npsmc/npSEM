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


    lag_x  = 5 # lag of removed analogs around x to avoid an over-fitting forecast
    lag_Dx = lambda Dx: np.shape(Dx)[-1]; # length of lag of outline analogs removed
    time_period = 1 # set 365.25 for year and 
    k_m    = [] # number of analogs 
    k_Q    = [] # number of analogs 
    nN_m   = np.arange(10,num_ana_m,20) #number of analogs to be chosen for mean estimation
    nN_Q   = np.arange(10,num_ana_Q,20) #number of analogs to be chosen for dynamical error covariance
    lag_k  = 1; #chosen the lag for k reestimation 
    k_lag  = 10
    k_inc  = 5
    estK   = 'same' # set 'same' if k_m = k_Q chosen, otherwise, set 'different'
    kernel = 'tricube'# set 'rectangular' or 'tricube'
    Q      = estQ;
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


def m_LLR(x,tx, ind_x,LLR):
    """ Apply the analog method on data of historical data to generate forecasts. """

    # initializations
    dx, N = x.shape;
    xf = np.zeros([dx,N]);
    mean_xf = np.zeros([dx,N]);
    Q_xf = np.zeros([dx,dx,N]);
    M_xf = np.zeros([dx+1,dx,N]);
    
    lag_x = LLR.lag_x; lag_Dx = LLR.lag_Dx(LLR.data.ana);
    if len(LLR.data.ana.shape)==1:
        dimx = 1; dimD =1; #lenC = LLR.data.ana.shape
    elif len(LLR.data.ana.shape)==2:
        dimD =1; dimx,_ = LLR.data.ana.shape
    else:
        dimx,dimD,_ = LLR.data.ana.shape;
#    TC = lenC +1;
    try:
        indCV =  (np.abs((tx-LLR.data.time)) % LLR.time_period <= lag_Dx) & (np.abs(tx-LLR.data.time) >= lag_x)
    except:
        indCV =  (np.abs(tx-LLR.data.time) >= lag_x)

    lenD = np.shape(LLR.data.ana[...,np.squeeze(indCV)])[-1]
    analogs_CV = np.reshape(LLR.data.ana[...,np.squeeze(indCV)],(dimx,lenD*dimD));
    successors_CV = np.reshape(LLR.data.suc[...,np.squeeze(indCV)],(dimx,lenD*dimD));
 

    if (LLR.gam != 1):
        if len(LLR.data_prev.ana.shape)==1:
            dimx_prev = 1; dimD_prev =1; #lenC = LLR.data.ana.shape
        elif len(LLR.data_prev.ana.shape)==2:
            dimD_prev =1; dimx_prev,_ = LLR.data_prev.ana.shape
        else:
            dimx_prev,dimD_prev,_ = LLR.data_prev.ana.shape;
        indCV_prev = (indCV) & (LLR.data_prev.time==LLR.data_prev.time)
        lenD_prev = np.shape(LLR.data_prev.ana[...,indCV_prev])[-1]

        analogs_CV_prev = np.reshape(LLR.data_prev.ana[...,indCV_prev],(dimx_prev, lenD_prev*dimD_prev ));
        successors_CV_prev =np.reshape(LLR.data_prev.suc[...,indCV_prev],(dimx_prev, lenD_prev*dimD_prev ));     
        analogs = np.concatenate((analogs_CV,analogs_CV_prev),axis=1) ; successors = np.concatenate((successors_CV,successors_CV_prev),axis=1);
    else:
        analogs = analogs_CV; successors = successors_CV;

    #LLR.k_m = np.size(analogs,1)/np.size(analogs_CV,1)*LLR.LLR.k_m;
    #%% LLR estimating
    # #[ind_knn,dist_knn]=knnsearch(analogs,x,'k',LLR.k_m);
    LLR.k_m = min(LLR.k_m,np.size(analogs,1)); LLR.k_Q = min(LLR.k_m, LLR.k_Q);
    weights = np.ones((N,LLR.k_m))/LLR.k_m; # rectangular kernel for default
 
    for i in range(N):
      # search k-nearest neighbors
        X_i = np.tile(x[:,i],(np.size(analogs,1),1)).T;
        dist = np.sqrt(np.sum((X_i- analogs)**2,0));
        ind_dist = np.argsort(dist); 
        ind_knn = ind_dist[:LLR.k_m]

        if LLR.kernel == 'tricube':
            h_m = dist[ind_knn[-1]]; # chosen bandwidth to hold the constrain dist/h_m <= 1
            weights[i,:] = (1-(dist[ind_knn]/h_m)**3)**3;
      # identify which set analogs belong to (if using SAEM) 
        ind_prev = np.where(ind_knn>np.size(analogs_CV,1));
        ind = np.setdiff1d(np.arange(0,LLR.k_m),ind_prev);
        if (len(ind_prev) >0):
            weights[i,ind_prev] = (1-LLR.gam)*weights[i,ind_prev];
            weights[i,ind] = LLR.gam*weights[i,ind];

        wei  = weights[i,:]/np.sum(weights[i,:]);
        ## LLR coefficients
        W = np.sqrt(np.diag(wei));
        Aw = np.dot(np.insert(analogs[:,ind_knn],0,1,0),W)
        Bw = np.dot(successors[:,ind_knn],W)
        M = np.linalg.lstsq(Aw.T,Bw.T,rcond=-1)[0]; 
      # weighted mean and covariance
        mean_xf[:,i] = np.dot(np.insert(x[:,i],0,1,0),M);       
        M_xf[:,:,i] = M; 
        
        if (LLR.Q.type== 'adaptive'):
            res = successors[:,ind_knn]- np.dot(np.insert(analogs[:,ind_knn],0,1,0).T, M).T;
            
            if LLR.kernel == 'tricube':
                h_Q = dist[ind_knn[LLR.k_Q-1]]; # chosen bandwidth to hold the constrain dist/h_m <= 1
                wei_Q = (1-(dist[ind_knn[:LLR.k_Q]]/h_Q)**3)**3;
            else:
                wei_Q = wei[:LLR.k_Q];
            wei_Q = wei_Q/np.sum(wei_Q);

            cov_xf =  np.cov(res[:,:LLR.k_Q], aweights = wei_Q)#((res[:,:LLR.k_Q].dot(np.diag(wei_Q))).dot(res[:,:LLR.k_Q].T))/(1-sum(wei_Q**2));
            if (LLR.Q.form =='full'):
                Q_xf[:,:,i] = cov_xf;
            elif (LLR.Q.form =='diag'):
                Q_xf[:,:,i] = np.diag(np.diag(cov_xf));
            else:
                try:
                    Q_xf[:,:,i] = np.trace(cov_xf)*LLR.Q.base/dx;
                except:
                    Q_xf[:,:,i] = cov_xf*LLR.Q.base/dx;
       
        else:
            Q_xf[:,:,i] = LLR.Q.value;
        
    #%% LLR sampling 
    for i in range(N):
        if len(ind_x)>1:
            xf[:,i]  = np.random.multivariate_normal(mean_xf[:,ind_x[i]],Q_xf[:,:,ind_x[i]]);
        else:
            xf[:,i] = np.random.multivariate_normal(mean_xf[:,i],Q_xf[:,:,i]);
    
    return xf, mean_xf, Q_xf, M_xf; # end
        
            
            
        
        

            
        
    
    
