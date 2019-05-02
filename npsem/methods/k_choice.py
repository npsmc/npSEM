

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 13:21:13 2018

@author: trang
"""
import numpy as np
import matplotlib.pyplot as plt #plot
#from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
#from scipy.stats import multivariate_normal
from .llr_forecasting_cv import m_LLR

def k_choice(LLR, x, y, time):
    dx, N, T = x.shape;
    if (LLR.Q.type == 'fixed') or (LLR.estK == 'same'):
        L = np.zeros((len(LLR.nN_m),1));     E = np.zeros((len(LLR.nN_m),1))

        for i in range(len(LLR.nN_m)):
            LLR.k_m = LLR.nN_m[i]; LLR.k_Q = LLR.k_m;
            loglik = 0; err =0;
            for t in range(T):
                _, mean_xf, Q_xf, _=  m_LLR(x[...,t],time[t],np.ones([1]),LLR);
                innov = y[...,t]- mean_xf
                err += np.mean((innov)**2) 
            L[i] = loglik; E[i] = np.sqrt(err/T);
                
        ind_max = np.argmin(E); 
        k_m = LLR.nN_m[ind_max]; 
        k_Q = k_m
    else:
        X,Y = np.meshgrid(LLR.nN_m,LLR.nN_Q); Q = np.zeros((dx,dx,T));
        len_ana = len(LLR.nN_m)*len(LLR.nN_Q)
        L = np.zeros(len_ana); E = np.zeros(len_ana);

        for i in range(len_ana):
            LLR.k_m = np.squeeze(X.T.reshape(len_ana)[i]); 
            if  np.squeeze(Y.T.reshape(len_ana)[i]) > LLR.k_m:
                indY  =  divmod(i ,len(LLR.nN_Q));
                Y[indY[1],indY[0]] =   LLR.k_m; # condition: number of analogs for m estimates is always larger or equal to the one for Q estimates
            LLR.k_Q =np.squeeze(Y.T.reshape(len_ana)[i])
            loglik = 0; err=0
            for t in range(T):
                _, mean_xf, Q_xf, _=  m_LLR(x[...,t],time[t],np.ones([1]),LLR);
                innov = y[...,t]- mean_xf
                const = -.5 * np.log(2*np.pi*np.linalg.det(Q_xf.transpose(-1,0,1)))
                logwei = -.5*np.sum(innov.T.dot(np.linalg.inv(Q_xf.transpose(-1,0,1)))[np.arange(N),np.arange(N),:]*innov.T,1)
                loglik  +=  (np.sum(const + logwei))/N
                err += np.sqrt(np.mean((innov)**2)) 
 
                
            L[i] = loglik; E[i] = np.sqrt(err/T)
        ind_max= np.argmax(L)
        k_m = np.squeeze(X.T.reshape(len_ana)[ind_max]);
        k_Q = np.squeeze(Y.T.reshape(len_ana)[ind_max]);

    return k_m, k_Q
