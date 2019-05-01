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
from npsem.methods.generate_data import generate_data
from npsem.methods.llr_forecasting_cv import m_LLR
from npsem.methods.model_forecasting import m_true
from npsem.methods.k_choice import k_choice
from npsem.methods.cpf_bs_smoothing import _CPF_BS
from npsem.methods.sem import CPF_BS_SEM
from npsem.methods.npsem import LLR_CPF_BS_SEM
from npsem.methods.enks import _EnKS
from npsem.methods.additives import RMSE
from npsem.save_load import loadTr


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

a = 3;
mx = lambda x: np.sin(a*x) # fortran version (fast)
jac_mx = lambda x: a*np.cos(a*x) # python version (slow)

# Setting covariances
sig2_Q = 0.1; sig2_R = 0.1 # parameters
Q_true = np.eye(dx) *sig2_Q #  model variance
R_true = np.eye(dx) *sig2_R # observation variance

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
plt.draw()


# In[3]:


#%% FORECASTING
class estQ:
    value = Q_true;
    type = 'fixed' # chosen predefined type of model error variance ('fixed', 'adaptive')
    form = 'constant' # chosen esimated matrix form ('full', 'diag', 'constant')
    base =  np.eye(dx) # for fixed base of model variance (for 'constant' matrix form only)
    decision = True # chosen if Q is estimated or not ('True', 'False')
class estR:
    value = [];
    type = 'fixed' # chosen predefined type of observation error variance ('fixed', 'adaptive')
    form = 'constant' # chosen esimated matrix form ('full', 'diag', 'constant')
    base =  np.eye(dx) # for fixed base of model variance
    decision = True # chosen if R is estimated or not ('True', 'False')
class estX0:
    decision = False # chosen if X0 is estimated or not ('True', 'False')
    
class estD: # for non-parametric approach only
    decision = False # chosen if the smoothed data is updated or not ('True', 'False')

num_ana_m = 200;  num_ana_Q = 200
data_init = np.r_['0,2,0',Y_train.values[...,:-1], Y_train.values[...,1:]];
ind_nogap = np.where(~np.isnan(np.sum(data_init,0)))[0]; 
#  LLR_FORECASTING (non-parametric dynamical model constructed given the catalog)
# parameters of the analog forecasting method
class LLR:
    class data:
        ana = np.zeros((dx,1,len(ind_nogap))); suc = np.zeros((dx,1,len(ind_nogap)));
        ana[:,0,:] =data_init[:dx,ind_nogap]; suc[:,0,:]  = data_init[dx:,ind_nogap]; time = Y_train.time[ind_nogap] # catalog with analogs and successors

#        ana =data_init[:dx,ind_nogap]; suc = data_init[dx:,ind_nogap]; time = Y_train.time[ind_nogap] # catalog with analogs and successors
#        ana = X_train.values[...,:-1]; suc =X_train.values[...,1:]; time = X_train.time[:-1]#catalog with analogs and successors
    class data_prev:
        ana =data_init[:dx,ind_nogap]; suc = data_init[dx:,ind_nogap]; time = Y_train.time[ind_nogap] # catalog with analogs and successors
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


N_iter = 500# number of iterations of EM algorithms
# Step functions
gam1 = np.ones(N_iter,dtype=int)  #for SEM (stochastic EM)
gam2 = np.ones(N_iter, dtype=int)
for k in range(50,N_iter):
    gam2[k] = k**(-0.7) # for SAEM (stochastic approximation EM)
#gam1 = gam2

X_conditioning = np.zeros([dx,T_train]) # the conditioning trajectory (only for CPF-BS-SEM and CPF-AS-SEM)
B = np.ones([1,1])
xb= X_train.values[...,0]
# initial parameters
aintQ = 0.01; bintQ = .5
aintR = 0.01; bintR = .5
    
Nf = 10# number of particles
Ns = 5 # number of realizations
R_init = .5*np.eye(dy)    
Q_init = 3*np.eye(dx)
LLR.Q.value = Q_init

### (SEM): STOCHASTIC EXPECATION-MAXIMIZATION  vs (npSEM): NON-PARAMETRIC STOCHASTIC EXPECATION-MAXIMIZATION 

## SEM[m] 
m = lambda x,pos_x,ind_x,Q: m_true(x,pos_x,ind_x, Q, mx,jac_mx, dt_model) # true forecast model

# run EnKS for generating the fisrt conditioning trajectory 
m_init = lambda  x,pos_x,ind_x: m(x,pos_x,ind_x, Q_init)
Xs, _, _ = _EnKS(dx, 20, len(Y_train.time), H, R_init, Y_train.values,X_train.values, dy, xb, B, Q_init, 1, m_init)
X_conditioning = Xs.mean(1)

# outputs of SEM[m] algorithm
out_SEM = CPF_BS_SEM(Y_train.values,X_train.values, m, Q_init, H, R_init, xb , B, X_conditioning, dx, Nf, Ns, X_train.time,N_iter,gam1,estQ,estR,estX0)
  

## SEM[\hat m] 
estD.decision = False; 
LLR.Q.value = Q_init
LLR.data.ana = np.zeros((dx,1,T_train-1))
LLR.data.suc = np.zeros((dx,1,T_train-1))
LLR.data.ana[:,0,:] = X_train0.values[...,:-1]
LLR.data.suc[:,0,:] = X_train0.values[...,1:]
LLR.data.time = X_train.time[:-1]
k_m, k_Q = k_choice(LLR,LLR.data.ana,LLR.data.suc,LLR.data.time) # choose an optimal number of neighbors used in LLR forecast
LLR.k_m =k_m; LLR.k_Q =k_Q; 

LLR.lag_x =0; 
m_hat = lambda  x,pos_x,ind_x: m_LLR(x,pos_x,ind_x,LLR) # LLR forecast model
# run EnKS for generating the fisrt conditioning trajectory 
Xs, _, _ = _EnKS(dx, 20, len(Y_train.time), H, R_init, Y_train.values,
                 X_train.values, dy, xb, B, Q_init, 1, m_hat)
X_conditioning = Xs.mean(1)

# outputs of SEM[\hat m] algorithm
out_SEM_hat = LLR_CPF_BS_SEM(Y_train.values,X_train.values, 
              LLR, H, R_init,xb,B, X_conditioning,dx, Nf, 
              Ns,X_train.time, N_iter, gam1, estD, estQ, estR, estX0)
    
## npSEM   
estD.decision = True
LLR.lag_x =5
LLR.Q = estQ
LLR.nN_m = np.arange(10,num_ana_m,20)
LLR.nN_Q = np.arange(10,num_ana_Q,20)
LLR.data.ana = np.zeros((dx,1,len(ind_nogap)))
LLR.data.suc = np.zeros((dx,1,len(ind_nogap)))
LLR.data.ana[:,0,:] =data_init[:dx,ind_nogap]
LLR.data.suc[:,0,:] = data_init[dx:,ind_nogap]
LLR.data.time = Y_train.time[ind_nogap] 

LLR.Q.value = Q_init
k_m, k_Q = k_choice(LLR,LLR.data.ana,LLR.data.suc,LLR.data.time) # choose an optimal number of neighbors used in LLR forecast
LLR.k_m =k_m; LLR.k_Q =k_Q; 
    
m_hat = lambda  x,pos_x,ind_x: m_LLR(x,pos_x,ind_x,LLR) # LLR forecast model
# run EnKS for generating the fisrt conditioning trajectory 
Xs, _, _ = _EnKS(dx, 20, len(Y_train.time), H, R_init, 
                 Y_train.values,X_train.values, dy, xb, B, Q_init, 1, m_hat)
X_conditioning = Xs.mean(1)
    
# outputs of npSEM algorithm    
out_npSEM = LLR_CPF_BS_SEM(Y_train.values,X_train.values, LLR, H, R_init,xb,B, X_conditioning,dx, Nf, Ns,X_train.time, N_iter, gam1, estD, estQ, estR, estX0)

    


## results
ii =0; ilim =0
# SEM
Q_SEM = out_SEM['EM_state_error_covariance'][:,:,ii:]
R_SEM= out_SEM['EM_observation_error_covariance'][:,:,ii:]
loglik_SEM= out_SEM['loglikelihood'][ii:]
Xs_SEM = out_SEM['smoothed_samples']

Q_SEM_hat = out_SEM_hat['EM_state_error_covariance'][:,:,ii:]
R_SEM_hat = out_SEM_hat['EM_observation_error_covariance'][:,:,ii:]
loglik_SEM_hat= out_SEM_hat['loglikelihood'][ii:]
Xs_SEM_hat = out_SEM_hat['smoothed_samples']

Q_npSEM = out_npSEM['EM_state_error_covariance'][:,:,ii:]
R_npSEM= out_npSEM['EM_observation_error_covariance'][:,:,ii:]
loglik_npSEM= out_npSEM['loglikelihood'][ii:]
Xs_npSEM = out_npSEM['smoothed_samples']
k_opt = out_npSEM['optimal_number_analogs'][:,ii:]


# In[6]:



## results
ii =0; ilim =0
# SEM
Q_SEM = out_SEM['EM_state_error_covariance'][:,:,ii:]
R_SEM= out_SEM['EM_observation_error_covariance'][:,:,ii:]
loglik_SEM= out_SEM['loglikelihood'][ii:]
Xs_SEM = out_SEM['smoothed_samples']

Q_SEM_hat = out_SEM_hat['EM_state_error_covariance'][:,:,ii:]
R_SEM_hat = out_SEM_hat['EM_observation_error_covariance'][:,:,ii:]
loglik_SEM_hat= out_SEM_hat['loglikelihood'][ii:]
Xs_SEM_hat = out_SEM_hat['smoothed_samples']

Q_npSEM = out_npSEM['EM_state_error_covariance'][:,:,ii:]
R_npSEM= out_npSEM['EM_observation_error_covariance'][:,:,ii:]
loglik_npSEM= out_npSEM['loglikelihood'][ii:]
Xs_npSEM = out_npSEM['smoothed_samples']
k_opt = out_npSEM['optimal_number_analogs'][:,ii:]



### COMPUTE LOG-LIKELIHOOD RATIO STATISTICS AND ROOT OF MEAN SQUARE ERROR
ilim=N_iter; LLR.lag_x =5
T_SEM  = np.zeros([ilim])
T_SEM_hat  = np.zeros([ilim])
T_npSEM  = np.zeros([ilim])

RMSE_x_SEM  = np.zeros([ilim])
RMSE_x_SEM_hat  = np.zeros([ilim])
RMSE_x_npSEM  = np.zeros([ilim])
LLR.data.time = np.arange(T_train-1)

# true forecast
_, mean_xft,_, _ =  m(X_train.values[:,:-1],1,np.ones([1]), Q_true)
T_0 = np.sum(np.log(multivariate_normal.pdf(X_train.values[:,1:].T-mean_xft.T,np.zeros([dx]) ,Q_true)))

LLR.lag_x =5; LLR.nN_m = np.arange(10,num_ana_m,20) ; LLR.nN_Q = np.arange(10,num_ana_Q,20) 
LLR.data.ana = np.zeros((dx,1,T_train-1)); LLR.data.suc = np.zeros((dx,1,T_train-1));
LLR.data.ana[:,0,:] =X_train0.values[:,:-1]; LLR.data.suc[:,0,:]  = X_train0.values[:,1:]; LLR.data.time = X_train.time[:-1] # catalog with analogs and successors
k_m, k_Q = k_choice(LLR,LLR.data.ana,LLR.data.suc,LLR.data.time)
LLR.k_m=k_m; LLR.k_Q=k_Q; 
LLR.lag_x =0
xf, mean_xff, Q_xf, M_xf = m_LLR(X_train.values[:,:-1],1,np.ones([1]),LLR) 



for i in range(ilim):
    Q_i = Q_SEM[...,i]
    T_SEM[i] = np.sum(np.log(multivariate_normal.pdf(X_train.values[:,1:].T-mean_xft.T,np.zeros([dx]) ,Q_i)))
    RMSE_x_SEM[i]= RMSE(np.mean(Xs_SEM[:,:,:,i+1],1)-X_train.values)
  
    Q_xf_i = Q_SEM_hat[...,i]
    T_SEM_hat[i] = np.sum(np.log(multivariate_normal.pdf(X_train.values[:,1:].T-mean_xff.T,np.zeros([dx]) ,Q_xf_i)))
    RMSE_x_SEM_hat[i]= RMSE(np.mean(Xs_SEM_hat[:,:,:,i+1],1)-X_train.values)
   

    if i==0:
        LLR.lag_x =5
        LLR.data.ana = np.zeros((dx,1,len(ind_nogap))); LLR.data.suc = np.zeros((dx,1,len(ind_nogap)));
        LLR.data.ana[:,0,:] =data_init[:dx,ind_nogap]; LLR.data.suc[:,0,:]  = data_init[dx:,ind_nogap]; LLR.data.time = Y_train.time[ind_nogap]  # catalog with analogs and successors
        k_m, k_Q = k_choice(LLR,LLR.data.ana,LLR.data.suc,LLR.data.time)
        LLR.k_m=k_m; LLR.k_Q=k_Q; 
        
    else:
        LLR.data.ana= Xs_npSEM[:,:,:-1,i]; LLR.data.suc= Xs_npSEM[:,:,1:,i]; LLR.data.time = X_train.time[:-1]
        LLR.k_m = k_opt[0,i]; LLR.k_Q= LLR.k_m
    
    _, mean_xf_i, _, _ = m_LLR(X_train.values[:,:-1],1,np.ones([1]),LLR)
    Q_xf_i = Q_npSEM[...,i]
    T_npSEM[i] = np.sum(np.log(multivariate_normal.pdf(X_train.values[:,1:].T-mean_xf_i.T,np.zeros([dx]) ,Q_xf_i)))
    RMSE_x_npSEM[i]= RMSE(np.mean(Xs_npSEM[:,:,:,i+1],1)-X_train.values)


# In[7]:


# lim_max = max(np.max(np.trace(Q_npSEM_adap)/dx),np.max(np.trace(R_npSEM_adap)/dx))
# lim_min=0.05
# Plot of convergence results
print('Convergence results of SEM and npSEM algorithms')
plt.rcParams['figure.figsize'] = (15, 3) 
plt.rc('xtick', labelsize= 8) 
plt.rc('ytick', labelsize= 8)
plt.figure()
plt.subplot(141)
line1,=plt.plot((1,N_iter),(np.trace(Q_true)/dx,np.trace(Q_true)/dx),'k:')
line2,=plt.plot(np.trace(Q_SEM)/dx,color = 'k')
line3,=plt.plot(np.trace(Q_SEM_hat)/dx,color =  'b')
line4,=plt.plot(np.trace(Q_npSEM)/dx,color = 'r')
plt.xlabel('iteration($r$)')
plt.title('$Tr(Q)/3$')
plt.xlim([-5,ilim])
plt.grid()

# plot sig2_R estimates
plt.subplot(142)
line1,=plt.plot((1,N_iter),(np.trace(R_true)/dx,np.trace(R_true)/dx),'k:')
line2,=plt.plot(np.trace(R_SEM)/dy,color = 'k')
line3,=plt.plot(np.trace(R_SEM_hat)/dy,color =  'b')
line4,=plt.plot(np.trace(R_npSEM)/dx,color = 'r')
plt.xlabel('iteration($r$)')
plt.legend([line1,line2,line3,line4],['true','SEM ($m$)','SEM ($\hat m$)','npSEM'])
plt.title('$\sigma_{R_t}^2$-estimates')
plt.xlim([-5,ilim])

plt.grid()
plt.subplot(143)
line1,=plt.plot(-2*(T_0-T_SEM),color = 'k')
line2,=plt.plot(-2*(T_0-T_SEM_hat),color = 'b')
line3,=plt.plot(-2*(T_0-T_npSEM),color = 'r')
plt.xlabel('iteration($r$)')
plt.title('Likelihood-ratio statistics')
plt.xlim([-5,ilim])
plt.grid()

plt.subplot(144)
line1,=plt.plot(RMSE_x_SEM,color = 'k')
line2,=plt.plot(RMSE_x_SEM_hat,color = 'b')
line3,=plt.plot(RMSE_x_npSEM,color = 'r')
plt.xlabel('iteration($r$)')
plt.title('RMSE $(x_t,\hat x_t )$')
plt.xlim([-5,ilim])
plt.grid()
plt.show()

### 'Scattet plots of $(Y_{t-1}, Y_t)$ of the observations,  $(X_{t-1}, X_t)$ of the state and $(\~X_{t-1},\~ X_t)$ generated from npSEM algorithm
# And compare m-estimates learned on these data
state = np.ones([1, 3400])
state[0,:] = np.arange(-1.7,1.7,0.001)

# true m 
m = lambda x,pos_x,ind_x: m_true(x,pos_x,ind_x, Q_true, mx,jac_mx, dt_model)
_, mean_xft,_, _ =  m(state,1,np.ones([1]))

# hat m (LLR) learned on a "perfect" data derived from the state process (X_t)
LLR.data.ana = np.zeros((dx,1,T_train-1)); LLR.data.suc = np.zeros((dx,1,T_train-1));
LLR.data.ana[:,0,:] =X_train.values[:,:-1]; LLR.data.suc[:,0,:]  = X_train.values[:,1:]; LLR.data.time = X_train.time[:-1] # catalog with analogs and successors
LLR.lag_x =5; LLR.nN_m = np.arange(10,num_ana_m,20) ; LLR.nN_Q = np.arange(10,num_ana_Q,20) 
k_m, k_Q = k_choice(LLR,LLR.data.ana,LLR.data.suc,LLR.data.time)
LLR.lag_x =0;LLR.k_m =k_m; LLR.k_Q= k_Q
_, mean_xfp, _, _ = m_LLR(state,1,np.ones([1]),LLR)

# hat m (LLR) learned on a "noisy" data derived from the observation process (Y_t)
LLR.data.ana = np.zeros((dx,1,T_train-2)); LLR.data.suc = np.zeros((dx,1,T_train-2));
LLR.data.ana[:,0,:] =Y_train.values[:,:-1]; LLR.data.suc[:,0,:]  = Y_train.values[:,1:]; LLR.data.time = Y_train.time[:-1] # catalog with analogs and successors
LLR.lag_x =5; LLR.nN_m = np.arange(10,num_ana_m,20) ; LLR.nN_Q = np.arange(10,num_ana_Q,20) 
k_m, k_Q = k_choice(LLR,LLR.data.ana,LLR.data.suc,LLR.data.time)
LLR.lag_x =0;LLR.k_m =k_m; LLR.k_Q= k_Q
_, mean_xfn, _, _ = m_LLR(state,1,np.ones([1]),LLR)

# hat m (LLR) learned on a "corrected" data derived from the npSEM algorithm
LLR.data.time = np.arange(T_train-1)
LLR.data.ana= Xs_npSEM[:,:,:-1,-1]; LLR.data.suc= Xs_npSEM[:,:,1:,-1]
LLR.k_m = k_opt[0,-1]; LLR.k_Q= LLR.k_m
LLR.lag_x =0
_, mean_xf_npSEM, _, _ = m_LLR(state,1,np.ones([1]),LLR)
print('Compare m-estimates learned on simulated data of the state process, the observation process and the npSEM algorithm')
plt.rcParams['figure.figsize'] = (12, 3)               
plt.figure()
plt.subplot(131)
var = 0
plt.scatter(Y_train.values[var,:-1],Y_train.values[var,1:],color = 'k', s = .3)
plt.plot(state.T,mean_xft.T,'r')
plt.plot(state.T,mean_xfn.T,'b')
plt.xlabel('$Y_{t-1}$', fontsize=8)
plt.ylabel('$Y_{t}$', fontsize=8)
plt.xlim([-2.25,2.25])
plt.ylim([-2.25,2.25])
plt.title('observations', fontsize=8)
plt.grid()

plt.subplot(132)
plt.scatter(X_train.values[var,:-1],X_train.values[var,1:],color = 'k', s = .3)# [0.5,0.5,0.5]
plt.plot(state.T,mean_xft.T,'r',label= '$m$')
plt.plot(state.T,mean_xfp.T,'b', label = '$\widehat m$')
plt.legend( fontsize = 8, ncol =2)
plt.xlabel('$X_{t-1}$', fontsize=8)
plt.ylabel('$X_{t}$', fontsize=8)
plt.xlim([-2.25,2.25])
plt.ylim([-2.25,2.25])
plt.title('state', fontsize=8)
plt.tick_params(labelleft='off')
plt.grid()

plt.subplot(133)
plt.scatter(Xs_npSEM[var,0,:-1,-1],Xs_npSEM[var,0,1:,-1],color  = 'k', s = .3) 
plt.plot(state.T,mean_xft.T,'r')
plt.plot(state.T,mean_xf_npSEM.T,'b')
plt.xlim([-2.25,2.25])
plt.ylim([-2.25,2.25])
plt.title('npSEM', fontsize=8)
plt.tick_params(labelleft='off')
plt.xlabel('$\~X_{t-1}$')
plt.ylabel('$\~X_{t}$')
plt.grid()
plt.show()

# In[ ]:
