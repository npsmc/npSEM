#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 11:40:04 2018

@author: trang
"""

#!/usr/bin/env python

""" AnDA_data_assimilation.py: Apply stochastic and sequential data assimilation technics using model forecasting or analog forecasting. """

import numpy as np
from AnDA_codes.AnDA_stat_functions import resampleMultinomial, sample_discrete, sqrt_svd
from tqdm import tqdm

def data_assimilation(yo, DA):
    """ Apply stochastic and sequential data assimilation technics using model forecasting or analog forecasting. """

    # initializations
    n = len(DA.xb);
    T = yo.values.shape[0];
    class x_hat:
        part = np.zeros([T,DA.N,n]);
        weights = np.zeros([T,DA.N]);
        values = np.zeros([T,n]);
        cov_state = np.zeros([T,DA.N,n,n]);
        cov = np.zeros([T,DA.N,n,n]);
        time = yo.time;

#%%    

        # special case for k=1
        x_hat.pre_mean  = np.zeros([T,DA.N,n])
        for k in tqdm(range(0,T)):
           if k==0:
              xf=np.random.multivariate_normal(DA.xb,DA.B,DA.N); 
              xf_mean = np.repeat(DA.xb[np.newaxis],DA.N,0); x_hat.pre_mean[1,:,:] =  xf_mean;
              w_prev = 1/DA.N*np.ones(DA.N)
              
           else:
              '''
              N_eff = 1/np.sum(x_hat.weights[k-1,:]**x_hat.weights[k-1,:]);
              if N_eff <= DA.N/2:
                  indic=resampleMultinomial(x_hat.weights[k-1,:])
                  w_prev = 1/DA.N*np.ones(DA.N)
              else:
                  indic = 1:DA.N
                  w_prev = x_hat.weights[k-1,:]
              '''
              indic=resampleMultinomial(x_hat.weights[k-1,:])
              w_prev = 1/DA.N*np.ones(DA.N)
              xf, xf_mean, Q, M =DA.m(x_hat.part[k-1,:,:],indic);
              if len(Q.shape) ==3:
                  x_hat.cov_state[k,:,:,:] = Q
              else:
                  x_hat.cov_state[k,:,:,:] = np.reshape(np.tile(Q,(DA.N,1)),(DA.N,n,n))

              x_hat.pre_mean[k,:,:] = xf_mean;
           x_hat.part[k,:,:]=xf;
       # update 
           i_var_obs = np.where(~np.isnan(yo.values[k,:]))[0]; 
           if len(i_var_obs)>0:
                innov = np.tile(yo.values[k,i_var_obs],(DA.N,1)).T- DA.H[i_var_obs,:].dot(x_hat.part[k,:,:].T);
                logweights = -1/2*np.sum(innov.T.dot(np.linalg.inv(DA.R[np.ix_(i_var_obs,i_var_obs)]))*innov.T,1); # (up to an additive constant)
                logmax = max(logweights)
                weights_tmp = np.exp(logweights-logmax)*w_prev; # Subtract the maximum value for numerical stability
#               weights_tmp=mvnpdf(repmat(yo.values(k,i_var_obs),DA.N,1),(DA.H(i_var_obs,:)*xf')',DA.R(i_var_obs,i_var_obs));
       # normalization
                weights_tmp=weights_tmp/sum(weights_tmp);
               
           else:
                weights_tmp=1/DA.N*np.ones(DA.N)
#               ll(k) =0;
           x_hat.weights[k,:] = weights_tmp;
           x_hat.values[k,:] = np.sum(x_hat.part[k,:,:]*np.repeat(x_hat.weights[k,:][np.newaxis],n,0).T,0);
        if (DA.method == 'AnPS'):
           
           x_hat.FFpart = x_hat.part; x_hat.FFweights = x_hat.weights; x_hat.FFvalues = x_hat.values
           x_hat.part = np.zeros([T,DA.Ns,n]); x_hat.weights = np.zeros([T,DA.Ns])
           ind_smo = sample_discrete(x_hat.FFweights[-1,:],DA.Ns)
           #ind_smoo = range(Ns)
           x_hat.part[-1,:,:] = x_hat.FFpart[-1,ind_smo,:]
           for k in tqdm(range(T-2,-1,-1)):
               for i in range(DA.Ns):
                    innov = np.tile(x_hat.part[k+1,i,:], (DA.N,1))- x_hat.pre_mean[k+1,:,:]
                    logwei = -.5*np.sum(innov.dot(np.linalg.inv(x_hat.cov_state[k+1,:,:,:]))[np.arange(DA.N),np.arange(DA.N),:]*innov,1) #compute weights with adaptive variance
                    const = np.sqrt(2*np.pi*np.linalg.det(x_hat.cov_state[k+1,:,:,:]))
                    logmax = np.max(logwei)
                    wei_res = 1/const*np.exp(logwei -logmax)*x_hat.FFweights[k,:]
                    Ws = wei_res/np.sum(wei_res)
                    ind_smo[i] = sample_discrete(Ws,1)
               x_hat.part[k,:,:] = x_hat.FFpart[k,ind_smo,:]
           x_hat.weights =   1/DA.Ns*np.ones([T,DA.Ns])
           x_hat.values = x_hat.part.mean(1)
        # end AnPS
 #%%
    elif DA.method =='CPFopt_BS':
        # special case for k=1
        x_hat.pre_mean  = np.zeros([T,DA.N,n])
        for k in tqdm(range(0,T)):
           if k==0:
              xf=np.random.multivariate_normal(DA.xb,DA.B,DA.N); x_hat.cov_state[k,:,:,:] = np.reshape(np.tile(DA.B,(DA.N,1)),(DA.N,n,n));
              xf_mean = np.repeat(DA.xb[np.newaxis],DA.N,0); x_hat.pre_mean[1,:,:] =  xf_mean;
              Q =DA.B; sizeQ =1; indic = np.arange(DA.N)
              w_prev = 1/DA.N*np.ones(DA.N)
           else:
              '''
              N_eff = 1/np.sum(x_hat.weights[k-1,:]**x_hat.weights[k-1,:]);
              if N_eff <= DA.N/2:
                  indic=resampleMultinomial(x_hat.weights[k-1,:])
                  w_prev = 1/DA.N*np.ones(DA.N)
              else:
                  indic = 1:DA.N
                  w_prev = x_hat.weights[k-1,:]
              '''
              indic=resampleMultinomial(x_hat.weights[k-1,:])
              w_prev = 1/DA.N*np.ones(DA.N)
              x_hat.weights[k-1,:] = w_prev
              # forecast with p(x_t|x_{t-1})
              xf, xf_mean, Q, M =DA.m(x_hat.part[k-1,:,:],indic);
              if len(np.shape(Q)) ==3:
                  x_hat.cov_state[k,:,:,:] = Q; sizeQ = DA.N
              else:
                  x_hat.cov_state[k,:,:,:] = np.reshape(np.tile(Q,(DA.N,1)),(DA.N,n,n)); sizeQ = 1
              x_hat.pre_mean[k,:,:] = xf_mean;
              
               # forecast with p(x_t|x_{t-1},y_t)
           i_var_obs = np.where(~np.isnan(yo.values[k,:]))[0]; 
           
           if len(i_var_obs)>0:
                R = DA.R[np.ix_(i_var_obs,i_var_obs)]; H = DA.H[i_var_obs,:]; obs = yo.values[k,i_var_obs]
                sigX = np.linalg.inv( np.linalg.inv(Q) + np.squeeze(np.reshape(np.tile(H.T.dot(np.linalg.inv(R)).dot(H),(sizeQ,1)),(sizeQ,n,n))))
                
                
                innov = np.tile(obs,(DA.N,1)).T- H.dot(xf_mean[indic,:].T);
                if sizeQ>1:
                    mu = (xf_mean.dot(np.linalg.inv(Q))[np.arange(sizeQ),np.arange(sizeQ),:]+   H.T.dot(np.linalg.inv(R)).dot(np.tile(obs,(DA.N,1)).T).T).dot(sigX)[np.arange(sizeQ),np.arange(sizeQ),:]
                    x_hat.part[k,:,:] = mu[indic,:] +  np.random.multivariate_normal(np.zeros(n),np.eye(n),DA.N).dot(sqrt_svd(sigX[indic,:,:]))[np.arange(sizeQ),np.arange(sizeQ),:]
                    sigY = np.reshape(np.tile(R,(sizeQ,1)),(sizeQ,len(i_var_obs),len(i_var_obs))) + np.transpose(Q[indic,:,:].dot(H.T),(0,2,1)).dot(H.T)
                    
                    logweights = -0.5*np.sum(innov.T.dot(np.linalg.inv(sigY))[np.arange(sizeQ),np.arange(sizeQ),:]*innov.T,1)
                    const = np.sqrt(2*np.pi*np.linalg.det(sigY))
                    '''
                    logweights = np.zeros(DA.N); const = np.zeros(sizeQ)
                    for i in range(sizeQ):
                        logweights[i] = -1/2*innov[:,i].T.dot(np.linalg.inv(R+ H.dot(Q[indic[i],:,:]).dot(H.T))).dot(innov[:,i]); # (up to an additive constant)
                        const[i] = np.sqrt(2*np.pi*np.linalg.det(R+ H.dot(Q[indic[i],:,:]).dot(H.T)))
                    '''
                else:
                    mu = (xf_mean.dot(np.linalg.inv(Q))+   H.T.dot(np.linalg.inv(R)).dot(np.tile(obs,(DA.N,1)).T).T).dot(sigX)
                    x_hat.part[k,:,:] = mu[indic,:] +  np.random.multivariate_normal(np.zeros(n),np.eye(n),DA.N).dot(sqrt_svd(sigX))
                    logweights = -1/2*np.sum(innov.T.dot(np.linalg.inv(R+ H.dot(Q).dot(H.T)))*innov.T,1); # (up to an additive constant)
                    const = np.sqrt(2*np.pi*np.linalg.det(R+ H.dot(Q).dot(H.T)))
                logmax = max(logweights)
                weights_tmp = 1/const*np.exp(logweights-logmax)*w_prev; # Subtract the maximum value for numerical stability
#               weights_tmp=mvnpdf(repmat(yo.values(k,i_var_obs),DA.N,1),(DA.H(i_var_obs,:)*xf')',DA.R(i_var_obs,i_var_obs));
       # normalization
                weights_tmp=weights_tmp/sum(weights_tmp);
               
           else:
                x_hat.part[k,:,:]=xf
                weights_tmp=1/DA.N*np.ones(DA.N)
                
           x_hat.weights[k,:] = weights_tmp;
           x_hat.values[k,:] = np.sum(x_hat.part[k,:,:]*np.repeat(x_hat.weights[k,:][np.newaxis],n,0).T,0);
            
          
        if (DA.method == 'AnPSopt'):
           
           x_hat.FFpart = x_hat.part; x_hat.FFweights = x_hat.weights; x_hat.FFvalues = x_hat.values
           x_hat.part = np.zeros([T,DA.Ns,n]); x_hat.weights = np.zeros([T,DA.Ns])
           ind_smo = sample_discrete(x_hat.FFweights[-1,:],DA.Ns)
           #ind_smoo = range(Ns)
           x_hat.part[-1,:,:] = x_hat.FFpart[-1,ind_smo,:]
           for k in tqdm(range(T-2,-1,-1)):
               for i in range(DA.Ns):
                    innov = np.tile(x_hat.part[k+1,i,:], (DA.N,1))- x_hat.pre_mean[k+1,:,:]
                    logwei = -.5*np.sum(innov.dot(np.linalg.inv(x_hat.cov_state[k+1,:,:,:]))[np.arange(DA.N),np.arange(DA.N),:]*innov,1) #compute weights with adaptive variance
                    const = np.sqrt(2*np.pi*np.linalg.det(x_hat.cov_state[k+1,:,:,:]))
                    logmax = np.max(logwei)
                    wei_res = 1/const*np.exp(logwei -logmax)*x_hat.FFweights[k,:]
                    Ws = wei_res/np.sum(wei_res)
                    ind_smo[i] = sample_discrete(Ws,1)
               x_hat.part[k,:,:] = x_hat.FFpart[k,ind_smo,:]
           x_hat.weights =   1/DA.Ns*np.ones([T,DA.Ns])
           x_hat.values = x_hat.part.mean(1)
        # end AnPSopt         
        
    else :
        print("Error: choose DA.method between 'AnEnKF', 'AnEnKS', 'AnPF' ")
        quit()
    return x_hat         
  
        
            
            
