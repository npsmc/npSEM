#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 11:27:21 2018

@author: trang
"""
""" AnDA_analog_forecasting.py: Apply the analog method on data of historical data to generate forecasts. """

import numpy as np

def m_LLR(x,tx, ind_x,LLR):
    """ Apply the analog method on data of historical data to generate forecasts. """

    # initializations
    dx, N = x.shape;
    xf = np.zeros([dx,N]);
    mean_xf = np.zeros([dx,N]);
    Q_xf = np.zeros([dx,dx,N]);
    M_xf = np.zeros([dx+1,dx,N]);
    
    lag_x = LLR.lag_x
    lag_Dx = LLR.lag_Dx(LLR.data.ana);
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
            dimx_prev = 1
            dimD_prev =1 #lenC = LLR.data.ana.shape
        elif len(LLR.data_prev.ana.shape)==2:
            dimD_prev =1
            dimx_prev,_ = LLR.data_prev.ana.shape
        else:
            dimx_prev,dimD_prev,_ = LLR.data_prev.ana.shape
            
        indCV_prev = (indCV) & (LLR.data_prev.time==LLR.data_prev.time)
        lenD_prev = np.shape(LLR.data_prev.ana[...,indCV_prev])[-1]

        analogs_CV_prev = np.reshape(LLR.data_prev.ana[...,indCV_prev],(dimx_prev, lenD_prev*dimD_prev ));
        successors_CV_prev =np.reshape(LLR.data_prev.suc[...,indCV_prev],(dimx_prev, lenD_prev*dimD_prev ));     
        analogs = np.concatenate((analogs_CV,analogs_CV_prev),axis=1) 
        successors = np.concatenate((successors_CV,successors_CV_prev),axis=1)
    else:
        analogs = analogs_CV; successors = successors_CV

 
    LLR.k_m = min(LLR.k_m,np.size(analogs,1))
    LLR.k_Q = min(LLR.k_m, LLR.k_Q)
    weights = np.ones((N,LLR.k_m))/LLR.k_m # rectangular kernel for default
 
    for i in range(N):
      # search k-nearest neighbors
        X_i = np.tile(x[:,i],(np.size(analogs,1),1)).T
        dist = np.sqrt(np.sum((X_i- analogs)**2,0))
        ind_dist = np.argsort(dist)
        ind_knn = ind_dist[:LLR.k_m]

        if LLR.kernel == 'tricube':
            h_m = dist[ind_knn[-1]]; # chosen bandwidth to hold the constrain dist/h_m <= 1
            weights[i,:] = (1-(dist[ind_knn]/h_m)**3)**3
            
      # identify which set analogs belong to (if using SAEM) 
        ind_prev = np.where(ind_knn>np.size(analogs_CV,1))
        ind = np.setdiff1d(np.arange(0,LLR.k_m),ind_prev)
        if (len(ind_prev) >0):
            weights[i,ind_prev] = (1-LLR.gam)*weights[i,ind_prev]
            weights[i,ind] = LLR.gam*weights[i,ind]

        wei  = weights[i,:]/np.sum(weights[i,:]);
        ## LLR coefficients
        W = np.sqrt(np.diag(wei))
        Aw = np.dot(np.insert(analogs[:,ind_knn],0,1,0),W)
        Bw = np.dot(successors[:,ind_knn],W)
        M = np.linalg.lstsq(Aw.T,Bw.T)[0]
      # weighted mean and covariance
        mean_xf[:,i] = np.dot(np.insert(x[:,i],0,1,0),M)     
        M_xf[:,:,i] = M
        
        if (LLR.Q.type== 'adaptive')
            res = successors[:,ind_knn]-np.dot(np.insert(analogs[:,ind_knn],0,1,0).T, M).T
            
            if LLR.kernel == 'tricube':
                h_Q = dist[ind_knn[LLR.k_Q-1]] # chosen bandwidth to hold the constrain dist/h_m <= 1
                wei_Q = (1-(dist[ind_knn[:LLR.k_Q]]/h_Q)**3)**3
            else:
                wei_Q = wei[:LLR.k_Q]
            wei_Q = wei_Q/np.sum(wei_Q)

            cov_xf =  np.cov(res[:,:LLR.k_Q], aweights = wei_Q)
            if (LLR.Q.form =='full'):
                Q_xf[:,:,i] = cov_xf;
            elif (LLR.Q.form =='diag'):
                Q_xf[:,:,i] = np.diag(np.diag(cov_xf));
            else:
                Q_xf[:,:,i] = np.trace(cov_xf)*LLR.Q.base/dx;
       
        else:
            Q_xf[:,:,i] = LLR.Q.value;
        
    #%% LLR sampling 
    for i in range(N):
        if len(ind_x)>1:
            xf[:,i]  = np.random.multivariate_normal(mean_xf[:,ind_x[i]],Q_xf[:,:,ind_x[i]]);
        else:
            xf[:,i] = np.random.multivariate_normal(mean_xf[:,i],Q_xf[:,:,i]);
    
    return xf, mean_xf, Q_xf, M_xf; # end
        
            
            
        
        

            
        
    
    
