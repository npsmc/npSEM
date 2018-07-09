

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
from methods.LLR_forecasting_CV import m_LLR

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
                
#                _, mean_xf, Q_xf, _=  m_LLR(x[:,t][np.newaxis].T,time[t],np.ones([1]),LLR);
#                innov = y[:,t][np.newaxis].T- mean_xf
#                
#                loglik  += -.5 * np.log(2*np.pi*np.linalg.det(np.squeeze(Q_xf))) - .5 * innov.T.dot(np.linalg.inv(np.squeeze(Q_xf))).dot(innov)
#                err += np.sqrt(np.mean((y[:,t][np.newaxis].T- mean_xf)**2))

               
#                const = -.5 * np.log(2*np.pi*np.linalg.det(Q_xf.transpose(-1,0,1)))
#                logwei = -.5*np.sum(innov.T.dot(np.linalg.inv(Q_xf.transpose(-1,0,1)))[np.arange(N),np.arange(N),:]*innov.T,1)
#                
#                loglik  +=  (np.sum(const + logwei))/N
##                err += np.sqrt(np.mean((y[:,t][np.newaxis].T- mean_xf)**2))
                
            L[i] = loglik; E[i] = np.sqrt(err/T);
                
        ind_max = np.argmin(E); 
        k_m = LLR.nN_m[ind_max]; 
        k_Q = k_m
        plt.rcParams['figure.figsize'] = (8, 5)        
        plt.figure(3)
        fig, ax1 = plt.subplots()
        #ax2 = ax1.twinx()
        ax1.plot(LLR.nN_m, E, color='b')
        ax1.set_xlabel('number of $m$-analogs $(k_m)$')
#        ax1.set_ylabel('log likelihood')

#        ax2.plot(LLR.nN_m, E, color='b')
        ax1.set_ylabel('RMSE')
#        plt.grid()
        plt.show()
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
#                _, mean_xf, Q_xf, _=  m_LLR(x[:,t][np.newaxis].T,time[t],np.ones([1]),LLR);
#                innov = y[:,t][np.newaxis].T- mean_xf
#                loglik  += -.5 * np.log(2*np.pi*np.linalg.det(np.squeeze(Q_xf))) -.5 * innov.T.dot(np.linalg.inv(np.squeeze(Q_xf))).dot(innov)   
#                err += np.mean((y[:,t][np.newaxis].T- mean_xf)**2)
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
        print(L); print(E)
        LL = (L.reshape((len(LLR.nN_m),len(LLR.nN_Q)))).T
        plt.rcParams['figure.figsize'] = (9, 9)
        fig = plt.figure(3)

        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(X, Y, LL,cmap='Greys', alpha =0.75, linewidth=0.25,edgecolor='k', antialiased=False)
#        plt.plot(k_m,k_Q,max(L),'k*', markersize = 6)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        fig.colorbar(surf, shrink=0.5, aspect=5)
        ax.set_xlabel('$k_m$') #number of $m$-analogs
        ax.set_ylabel('$k_Q$') #number of $Q$-analogs
        ax.set_zlabel('log likelihood')
        plt.grid()
        plt.show()
        
        
        print('Q ={}'.format(np.mean(Q,2)))
#        fig = plt.figure(4)
#        plt.rcParams['figure.figsize'] = (5, 8)
#        ax = fig.gca(projection='3d')
#        surf = ax.plot_surface(X, Y, E.reshape((len(LLR.nN_m),len(LLR.nN_Q))), cmap='Blues',
#                       linewidth=0, antialiased=False)
#        ax.zaxis.set_major_locator(LinearLocator(10))
#        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
#        fig.colorbar(surf, shrink=0.5, aspect=5)
#        ax.set_xlabel('number of analogs $(k_m)$')
#        ax.set_ylabel('number of analogs $(k_Q)$')
#        ax.set_zlabel('RMSE')
#        plt.grid()
#        plt.show()

        
#        fig = plt.figure(4)
#        plt.rcParams['figure.figsize'] = (5, 8)       
# Add a color bar which maps values to colors.

    return k_m, k_Q