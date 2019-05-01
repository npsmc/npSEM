#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 17:56:42 2018

@author: trang
"""
import numpy as np
import matplotlib.pyplot as plt #plot
import seaborn as sns
from numpy.linalg import inv
from scipy.linalg import sqrtm
from .llr_forecasting_cv import m_LLR
from .k_choice import k_choice
from .cpf_bs_smoothing import _CPF_BS
from .additives import RMSE
from tqdm import tqdm

def maximize(Xs,Xf_mean, y, H, estQ, estR):
    dx, Ns, T = Xs.shape
    xb = np.mean(Xs[:,:,0], 1)
    B = np.cov(Xs[:,:,0])
    Q= []; R= []
  
    if estQ.type == 'fixed':
        SigQ = np.zeros((dx, Ns, T-1))
        for t in range(T-1):
            SigQ[...,t] = Xs[...,t+1] - Xf_mean[...,t+1]
        SigQ = np.reshape(SigQ, (dx, (T-1)*Ns))
        SigQ = SigQ.dot(SigQ.T) / ((T-1)*Ns)
        if estQ.form == 'full':
            Q = SigQ
        elif estQ.form == 'diag':
            Q = np.diag(np.diag(SigQ))
        else:
            Q = estQ.base*np.trace(np.linalg.inv(estQ.base).dot(SigQ)) / (dx)

  
    nobs = 0; 
    if estR.type == 'fixed':
        var_obs = np.where(~np.isnan(y[:,0]))[0]; dy = len(var_obs)
        SigR = np.zeros([dy, Ns, T-1]); R = np.zeros([dx,dx]);
        for t in range(T-1):
            if dy>0:
                nobs += 1
                SigR[:,:,t] = np.tile(y[var_obs,t], (Ns, 1)).T - H[var_obs,:].dot(Xs[...,t+1])
        SigR = np.reshape(SigR, (dy, (T-1)*Ns))
        Robs = SigR.dot(SigR.T) / (nobs*Ns)
        if estR.form == 'full':
            R[np.ix_(var_obs,var_obs)] = Robs
        elif estR.form == 'diag':
            R[np.ix_(var_obs,var_obs)] = np.diag(np.diag(Robs))
        else:
            R[np.ix_(var_obs,var_obs)] = estR.base[np.ix_(var_obs,var_obs)]*np.trace(inv(estR.base[np.ix_(var_obs,var_obs)]).dot(Robs))/dy
    else:
        SigR =0; dy  = np.zeros(T-1,dtype = int)
        for t in range(T-1):
            var_obs = np.where(~np.isnan(y[:,t]))[0]; 
            dy[t] = len(var_obs)
            if dy[t]>0:
                innov = np.tile(y[var_obs,t], (Ns, 1)).T -H[var_obs,:].dot(Xs[...,t+1])
#                SigR += np.sum((np.linalg.norm(np.dot(sqrtm(inv(estR.base[np.ix_(var_obs,var_obs)])),innov), axis = 0))**2)
                SigR += np.sum((innov.T.dot(inv(estR.base[np.ix_(var_obs,var_obs)])))*innov.T)

        R = estR.base*SigR/(np.sum(dy)*Ns)
        
    return xb, B, Q, R  



def LLR_CPF_BS_SEM(Y,X,LLR,H,R,xb,B,Xcond,dx, Nf, Ns,time,nIter,gam_SAEM,estD,estQ,estR,estX0):
    
    T  = len(time)-1
    loglik = np.zeros(nIter)
    rmse_em = np.zeros(nIter)
    k_all  = np.zeros([2, nIter+1],dtype= int)
    Q_all  = np.zeros(np.r_[LLR.Q.value.shape,  nIter+1])
    R_all  = np.zeros(np.r_[R.shape,  nIter+1])
    B_all  = np.zeros(np.r_[B.shape,  nIter+1])
    xb_all = np.zeros(np.r_[xb.shape, nIter+1])
    Xcond_all = np.zeros([dx,T+1,nIter+1])
    Xs_all = np.zeros([dx,Ns,T+1,nIter+1])
    
    Q_all[:,:,0] = LLR.Q.value
    R_all[:,:,0] = R
    xb_all[:,0]  = xb
    B_all[:,:,0] = B
    Xcond_all[:,:,0] = Xcond
    k_all[0,0] = LLR.k_m ; k_all[1,0] = LLR.k_Q 
#   Xcond = np.zeros([dx,T])
    for r in tqdm(range(nIter)):
        # LLR-forecasting setting 
        m = lambda x,pos_x,ind_x: m_LLR(x,pos_x,ind_x,LLR)

        ## Expectation (E)-step:
        Xs, Xa, Xf, Xf_mean, Xf_cov, llh = _CPF_BS(Y,X,m,LLR.Q.value,H,R,xb,B,Xcond,dx, Nf, Ns,time[1:])
        loglik[r] = llh#log_likelihood(Xf, Y, H, R)
        rmse_em[r] = RMSE(X[:,1:] - Xs[...,1:].mean(1))
        Xcond = Xs[:,-1,:]
        ## Maximization (M)-step:
        xb_new, B_new, Q_new, R_new = maximize(Xs, Xf_mean, Y, H, estQ, estR)
       
        if estQ.decision:
            if estQ.type == 'fixed':
                LLR.Q.value = Q_new
            else:
                lenQ = len(np.shape(Xf_cov))
                LLR.Q.value = np.squeeze(np.mean(np.mean(Xf_cov,lenQ-1),lenQ-2))
        if estR.decision:
            R = R_new
        if estX0.decision:
            xb = xb_new; B = B_new
                
        gam = gam_SAEM[r]
        Q_all[...,r+1] = gam*LLR.Q.value + (1-gam)*Q_all[...,r] 
        R_all[...,r+1] = gam*R + (1-gam)*R_all[...,r] 
        xb_all[...,r+1] = gam*xb + (1-gam)*xb_all[...,r] 
        B_all[...,r+1] = gam*B + (1-gam)*B_all[...,r] 

        ## Additional (A)-step:
        if estD.decision:
            LLR.data_prev.ana = Xs_all[:,:-1,:-1,r]; LLR.data_prev.suc = Xs_all[:,:-1,1:,r]; LLR.data_prev.time = time[:-1]
            LLR.gam = gam
            LLR.data.ana  = Xs[...,:-1];  LLR.data.suc  =Xs[...,1:]; LLR.data.time = time[:-1]
            LLR.gam = gam
            if (r<2) or ((r % LLR.lag_k)==0):
                if r==0:
                    LLR.nN_m = np.arange(max(Ns*k_all[0,r]-LLR.k_lag,10),Ns*k_all[0,r]+LLR.k_lag,LLR.k_inc)
                    LLR.nN_Q = np.arange(max(Ns*k_all[1,r]-LLR.k_lag,10),Ns*k_all[1,r]+LLR.k_lag,LLR.k_inc)
                else:
                    LLR.nN_m = np.arange(max(k_all[0,r]-LLR.k_lag,10),k_all[0,r]+LLR.k_lag,LLR.k_inc)
                    LLR.nN_Q = np.arange(max(k_all[1,r]-LLR.k_lag,10),k_all[1,r]+LLR.k_lag,LLR.k_inc)
                km, kQ = k_choice(LLR, LLR.data.ana,LLR.data.suc,time[:-1])
                LLR.k_m =km; LLR.k_Q = kQ

        k_all[0,r+1] = LLR.k_m ; k_all[1,r+1] = LLR.k_Q 
        Xs_all[...,r+1] = Xs
        Xcond_all[...,r+1] = Xcond


    out_SEM = {'EM_LLR'                         : LLR,
              'smoothed_samples'               : Xs_all,
              'conditioning_trajectory'        : Xcond_all,
              'optimal_number_analogs'         : k_all,  
              'EM_x0_mean'                     : xb_all,
              'EM_x0_covariance'               : B_all,
              'EM_state_error_covariance'      : Q_all,
              'EM_observation_error_covariance': R_all,
              'loglikelihood'                  : loglik,
              'RMSE'                           : rmse_em}
             
    return out_SEM

