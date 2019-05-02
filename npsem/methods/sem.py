#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 17:58:02 2018

@author: trang
"""

"""
@author: Thi Tuyet Trang Chau

"""
import seaborn as sns
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt #plot
from methods.CPF_BS_smoothing import _CPF_BS
from methods.additives import RMSE
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
        SigR = np.zeros([dy, Ns, T-1]); R = np.nan*np.zeros([dx,dx]);
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



def CPF_BS_SEM(Y,X,m_true,Q,H,R,xb,B,Xcond,dx, Nf, Ns,time,nIter,gam_SAEM,estQ,estR,estX0):
        
    T  = len(time)-1
    loglik = np.zeros(nIter)
    rmse_em = np.zeros(nIter)
    T_em = np.zeros(nIter)
    Q_all  = np.zeros(np.r_[Q.shape,  nIter+1])
    R_all  = np.zeros(np.r_[R.shape,  nIter+1])
    B_all  = np.zeros(np.r_[B.shape,  nIter+1])
    xb_all = np.zeros(np.r_[xb.shape, nIter+1])
    Xcond_all = np.zeros([dx,T+1,nIter+1])
    Xs_all = np.zeros([dx,Ns,T+1,nIter+1])
    
    
    Q_all[:,:,0] = Q
    R_all[:,:,0] = R
    xb_all[:,0]  = xb
    B_all[:,:,0] = B
    Xcond_all[:,:,0] = Xcond
    for r in tqdm(range(nIter)):
        #  true-forecasting setting 
        m = lambda x,pos_x,ind_x: m_true(x,pos_x,ind_x,Q)

       
        # Expectation (E)-step
        Xs, Xa, Xf, Xf_mean,_,llh = _CPF_BS(Y,X,m,Q,H,R,xb,B,Xcond,dx, Nf, Ns,time[1:])
        loglik[r] = llh#log_likelihood(Xf, Y, H, R)
        rmse_em[r] = RMSE(X[:,1:] - Xs[...,1:].mean(1))
        Xcond = Xs[:,-1,:]
        # Maximization (M)-step
        xb_new, B_new, Q_new, R_new = maximize(Xs, Xf_mean, Y, H, estQ, estR)
        
        if estQ.decision:
            Q = Q_new
        if estR.decision:
            R = R_new
        if estX0.decision:
            xb = xb_new; B = B_new
                
        gam = gam_SAEM[r]
        Q_all[...,r+1] = gam*Q + (1-gam)*Q_all[...,r] 
        R_all[...,r+1] = gam*R + (1-gam)*R_all[...,r] 
        xb_all[...,r+1] = gam*xb + (1-gam)*xb_all[...,r] 
        B_all[...,r+1] = gam*B + (1-gam)*B_all[...,r] 

        Xs_all[...,r+1] = Xs
        Xcond_all[...,r+1] = Xcond
        Xs_all[...,r+1] = Xs
        Xcond_all[...,r+1] = Xcond
  
        
    out_SEM = {'smoothed_samples'               : Xs_all,
              'conditioning_trajectory'        : Xcond_all,
              'EM_x0_mean'                     : xb_all,
              'EM_x0_covariance'               : B_all,
              'EM_state_error_covariance'      : Q_all,
              'EM_observation_error_covariance': R_all,
              'loglikelihood'                  : loglik,
              'RMSE'                           : rmse_em}
             
    return out_SEM

