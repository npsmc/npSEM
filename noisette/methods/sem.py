#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 17:58:02 2018

@author: trang
"""

"""
@author: Thi Tuyet Trang Chau

"""
import numpy as np
from methods.CPF_BS_smoothing import _CPF_BS
from methods.additives import RMSE
from numpy.linalg import inv
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
    cp_em = np.zeros((dx,nIter))
    #corr_em = np.zeros((dx,nIter+1))
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
#   Xcond = np.zeros([dx,T])
    for r in tqdm(range(nIter)):
        #  true-forecasting setting 
        m = lambda x,pos_x,ind_x: m_true(x,pos_x,ind_x,Q)

        
        # Expectation (E)-step
        Xs, Xa, Xf, Xf_mean,_,llh = _CPF_BS(Y,X,m,Q,H,R,xb,B,Xcond,dx, Nf, Ns,time[1:])
        loglik[r] = llh#log_likelihood(Xf, Y, H, R)
        rmse_em[r] = RMSE(X[:,1:] - Xs[...,1:].mean(1))
#        cp_em[:,r],_,_ = CP95(Xs[...,1:],X[...,1:])[0]    
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
#        sns.set_style("whitegrid")        
#        plt.rcParams['figure.figsize'] = (12, 4)
#        plt.figure(1)
#        plt.plot(X[:,1:].T,color='grey', linewidth = 2)
#        plt.plot(Xcond[:,1:].T,'r-', linewidth = 2)
#        plt.plot(Y.T,'k.')
#        plt.xlabel('times')
#        plt.title('true data (curve), artificial data(point)')
#
#        
#        plt.rcParams['figure.figsize'] = (12, 12)           
#        plt.figure(5)
#        plt.subplot(131)
#        var = np.arange(0,dx)
#
#     
#        plt.scatter(Y[var,0:-2],Y[var,1:-1],color ='b')
#        plt.scatter(X[var,0:-2],X[var,1:-1],color ='k')  
#        plt.scatter(Xs[var,:,0:-2],Xs[var,:,1:-1],color ='r')   
#        plt.xlim([-20,20]); plt.ylim([-20,20])
#        plt.title('data')
#        plt.xlabel('analogs')
#        plt.ylabel('successors')
#        plt.grid()
#        
#        plt.subplot(132)
#        plt.scatter(Xs[var,1,1:],X[var,1:],color ='k')        
#        plt.xlim([-20,20]); plt.ylim([-20,20])
#        plt.xlabel('reconstructed data')
#        plt.ylabel('true data')        
#        plt.grid()
#
#        plt.subplot(133)
#        plt.scatter(Y[var,:],X[var,1:],color ='k')
#        plt.xlim([-20,20]); plt.ylim([-20,20])
#        plt.xlabel('noisy data')
#        plt.ylabel('true data')  
#        plt.grid()
#        plt.show()
#
#        plt.figure(6)
#        plt.subplot(331)
#        var = 0
#        plt.scatter(Y[var,:-1],Y[var,1:],color = [0,0,0], s = 5)
#        plt.xlim([-25,25]); plt.ylim([-25,25])
#        plt.title('observed data')
#        plt.xlabel('analogs')
#        plt.ylabel('successors')
#        plt.grid()
#        
#        plt.subplot(332)
#        plt.scatter(Xs[var,:,:-1],Xs[var,:,1:], color =[0.25,0.25,0.25], s = 5)  
#        plt.xlim([-25,25]); plt.ylim([-25,25])
#        plt.title('recontructed data')
#        plt.xlabel('analogs')
#        plt.ylabel('successors')
#        plt.grid()
#        
#        plt.subplot(333)
#        var = 0
#        plt.scatter(X[var,:-1],X[var,1:],color = [0.5,0.5,0.5], s = 5) 
#        plt.xlim([-25,25]); plt.ylim([-25,25])
#        plt.title('true data')
#        plt.xlabel('analogs')
#        plt.ylabel('successors')
#        plt.grid()
#        
#        plt.subplot(334)
#        var = 1
#        plt.scatter(Y[var,:-1],Y[var,1:],color = [0,0,0], s = 5)
#        plt.xlim([-30,30]); plt.ylim([-30,30])
#        plt.xlabel('analogs')
#        plt.ylabel('successors')
#        plt.grid()
#        
#        plt.subplot(335)
#        plt.scatter(Xs[var,:,:-1],Xs[var,:,1:], color =[0.25,0.25,0.25], s = 5) 
#        plt.xlim([-30,30]); plt.ylim([-30,30])
#        plt.xlabel('analogs')
#        plt.ylabel('successors')
#        plt.grid()
#        
#        plt.subplot(336)
#        plt.scatter(X[var,:-1],X[var,1:],color = [0.5,0.5,0.5], s = 5)  
#        plt.xlim([-30,30]); plt.ylim([-30,30])
#        plt.xlabel('analogs')
#        plt.ylabel('successors')
#        plt.grid()
#        
#        plt.subplot(337)
#        var =2
#        plt.scatter(Y[var,:-1],Y[var,1:],color = [0,0,0], s = 5)
#        plt.xlim([0,50]); plt.ylim([0,50])
#        plt.xlabel('analogs')
#        plt.ylabel('successors')
#        plt.grid()
#        
#        plt.subplot(338)
#        plt.scatter(Xs[var,:,:-1],Xs[var,:,1:], color =[0.25,0.25,0.25], s = 5)  
#        plt.xlim([0,50]); plt.ylim([0,50])
#        plt.xlabel('analogs')
#        plt.ylabel('successors')
#        plt.grid()
#        
#        plt.subplot(339)
#        var = 2
#        plt.scatter(X[var,:-1],X[var,1:],color = [0.5,0.5,0.5], s = 5) 
#        plt.xlim([0,50]); plt.ylim([0,50])
#        plt.xlabel('analogs')
#        plt.ylabel('successors')
#        plt.grid()
#        plt.show()
#
#       
##        fig=plt.figure(6)
##
##        plt.rc('xtick', labelsize= 6) 
##        plt.rc('ytick', labelsize= 6)
##        ax=fig.gca(projection='3d')
##        line2,=ax.plot(np.squeeze(Xs[0,-1,1:]),np.squeeze(Xs[1,-1,1:]),np.squeeze(Xs[2,-1,1:]),'r',linewidth =1)
##        line0,=ax.plot(Y[0,...],Y[1,...],Y[2,...],'.k',markersize=3)
##        line1,=ax.plot(X[0,1:],X[1,1:],X[2,1:],'k',linewidth =1)
##        ax.set_xlabel('$x_1$',fontsize=7);ax.set_ylabel('$x_2$',fontsize=7);ax.set_zlabel('$x_3$',fontsize=7)
##        plt.legend([line0, line1, line2], ['observation','true state', 'smoothed mean'],bbox_to_anchor=(.5, .92), loc=2, borderaxespad=0., fontsize = 6)
##        plt.title('Smoothing with true error covariances $(Q= 0.01I_3, R = 2I_3)$', fontsize = 8)
#                
#        print('Q= {}, R = {}'.format(Q,R))
#        plt.rcParams['figure.figsize'] = (12, 6)
#        plt.figure(7)
#        plt.subplot(121)
#        sns.heatmap(Q, annot=True,cmap ="Greys")    
#        plt.title('Q estimates')
#
#        plt.subplot(122)
#        sns.heatmap(R, annot=True, cmap ="Greys")    
#        plt.title('R estimates')
#
#
#        plt.rcParams['figure.figsize'] = (12, 12) 
#        plt.figure(8)
#        plt.subplot(331)
#        var = 0
#        plt.plot(Y[var,:-1],Y[var,1:],color = [0,0,0])
#        plt.xlim([-25,25]); plt.ylim([-25,25])
#        plt.title('observed data')
#        plt.xlabel('analogs')
#        plt.ylabel('successors')
#        plt.grid()
#        
#        plt.subplot(332)
#        plt.plot(Xs[var,0,:-1],Xs[var,0,1:], color =[0.25,0.25,0.25])  
#        plt.xlim([-25,25]); plt.ylim([-25,25])
#        plt.title('recontructed data')
#        plt.xlabel('analogs')
#        plt.ylabel('successors')
#        plt.grid()
#        
#        plt.subplot(333)
#        var = 0
#        plt.plot(X[var,:-1],X[var,1:],color = [0.5,0.5,0.5]) 
#        plt.xlim([-25,25]); plt.ylim([-25,25])
#        plt.title('true data')
#        plt.xlabel('analogs')
#        plt.ylabel('successors')
#        plt.grid()
#        
#        plt.subplot(334)
#        var = 1
#        plt.plot(Y[var,:-1],Y[var,1:],color = [0,0,0]) 
#        plt.xlim([-30,30]); plt.ylim([-30,30])
#        plt.xlabel('analogs')
#        plt.ylabel('successors')
#        plt.grid()
#        
#        plt.subplot(335)
#        plt.plot(Xs[var,0,:-1],Xs[var,0,1:], color =[0.25,0.25,0.25])  
#        plt.xlim([-30,30]); plt.ylim([-30,30])
#        plt.xlabel('analogs')
#        plt.ylabel('successors')
#        plt.grid()
#        
#        plt.subplot(336)
#        plt.plot(X[var,:-1],X[var,1:],color = [0.5,0.5,0.5]) 
#        plt.xlim([-30,30]); plt.ylim([-30,30])
#        plt.xlabel('analogs')
#        plt.ylabel('successors')
#        plt.grid()
#        
#        plt.subplot(337)
#        var =2
#        plt.plot(Y[var,:-1],Y[var,1:],color = [0,0,0]) 
#        plt.xlim([0,50]); plt.ylim([0,50])
#        plt.xlabel('analogs')
#        plt.ylabel('successors')
#        plt.grid()
#        
#        plt.subplot(338)
#        plt.plot(Xs[var,0,:-1],Xs[var,0,1:], color =[0.25,0.25,0.25]) 
#        plt.xlim([0,50]); plt.ylim([0,50])
#        plt.xlabel('analogs')
#        plt.ylabel('successors')
#        plt.grid()
#        
#        plt.subplot(339)
#        var = 2
#        plt.plot(X[var,:-1],X[var,1:],color = [0.5,0.5,0.5]) 
#        plt.xlim([0,50]); plt.ylim([0,50])
#        plt.xlabel('analogs')
#        plt.ylabel('successors')
#        plt.grid()
#        plt.show()
        
#        plt.rcParams['figure.figsize'] = (12, 15)
#        plt.figure(8)
#        plt.subplot(311)
#        sns.heatmap(Y,cmap ="Greys")    
#        plt.title('observed data')
#        plt.subplot(312)
#        sns.heatmap(X[:,1:],cmap ="Greys")    
#        plt.title('true data')        
#        plt.subplot(313)
#        sns.heatmap(np.squeeze(Xs.mean(1))[:,1:],cmap ="Greys")    
#        plt.title('smoothed data')       
        
    out_SEM = {'smoothed_samples'               : Xs_all,
              'conditioning_trajectory'        : Xcond_all,
              'EM_x0_mean'                     : xb_all,
              'EM_x0_covariance'               : B_all,
              'EM_state_error_covariance'      : Q_all,
              'EM_observation_error_covariance': R_all,
              'loglikelihood'                  : loglik,
              'RMSE'                           : rmse_em,
              'coverage_probability'           : cp_em}
             
    return out_SEM

