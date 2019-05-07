# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 13:21:13 2018

@author: trang

modified by Valerie (2019/05/05)
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LinearLocator, FormatStrFormatter


class Est:
    """
     type: chosen predefined type of model error covatiance ('fixed', 'adaptive')
     form: chosen esimated matrix form ('full', 'diag', 'constant')
     base: for fixed base of model covariance (for 'constant' matrix form only)
     decision: chosen if Q is estimated or not ('True', 'False')
    """

    def __init__(self, value=None, type=None, form=None, base=None, decision=None):
        self.value = value
        self.type = type
        self.form = form
        self.base = base
        self.decision = decision


class Data:

    def __init__(self, data_init, time, ind_nogap):
        dx = data_init.shape[0]
        self.analogs = np.zeros((dx, 1, len(ind_nogap)-1))
        self.succesors = np.zeros((dx, 1, len(ind_nogap)-1))
        T = data_init.shape[1] # a voir format data_init?
        self.analogs[:, 0, :] = data_init[:, ind_nogap[:T-1]]
        self.succesors[:, 0, :] = data_init[:, ind_nogap[1:]]
        self.time = time
        self.L = self.analogs.shape[2] # lenght of analogs time series

# parameters of the analog forecasting method
class AF:
    def __init__(self,catalog,k=50,regression="locally_constant",sampling="gaussian",
                 kernel="tricube",lambdaa=1,lag_x=0):
        self.k = k # number of analogs
        self.neighborhood = None#np.ones([xt.values.shape[1],xt.values.shape[1]]) # global analogs
        self.catalog = catalog # catalog with analogs and successors
        self.regression = regression # chosen regression ('locally_constant', 'increment', 'local_linear')
        self.sampling = sampling  # chosen sampler ('gaussian', 'multinomial')
        self.lag_x = lag_x
        self.kernel = kernel
        self.lambdaa = lambdaa # kernel bandwidth

    def m_LLR(self, x, tx, ind_x):
        # Local Linear Regression with kernel
        # No cross validation -> fixed number of neighbors and bandwidth
        # x: set of particles
        # tx: current time associate to the set of particles
        # ind_x: number of particles to return par input particle
        
        # initializations
        dx, N = x.shape
        xf = np.zeros([dx, N])
        mean_xf = np.zeros([dx, N])
        Q_xf = np.zeros([dx, dx, N])
        M_xf = np.zeros([dx + 1, dx, N])
        
        T = self.catalog.L
        ind_train = np.delete(np.arange(T),np.arange(max(tx-self.lag_x,0),min(tx+self.lag_x+1,T)))
        #ind_train=np.arange(T)
        N_train = self.catalog.analogs.shape[1] # number of particles in the train dataset
        X_train = self.catalog.analogs[:,:,ind_train].reshape((dx,N_train*len(ind_train))).T
        y_train = self.catalog.succesors[:,:,ind_train].reshape((dx,N_train*len(ind_train))).T
        X_test = x.T
        
        if self.kernel=='tricube':
            def wght(x):
                return (1 - (x / self.lambdaa) ** 3) ** 3
        else:
            def wght(x):
                return 1/len(x)    
        neigh = NearestNeighbors(n_neighbors=self.k) 
        neigh.fit(X_train) 
        Xn = neigh.kneighbors(X_test) # return_distance=True (default)
        y_llr= np.zeros(X_test.shape)
        lr = LinearRegression()
        Q_xf=np.zeros((dx,dx,N))
        for i in range(N):
            i_neigh = Xn[1][i]
            lr.fit(X_train[i_neigh,:],y_train[i_neigh,:],sample_weight=wght(Xn[0][i])) # add a weight!
            y_llr[i,:] = lr.predict(X_test[i,:].reshape(1,X_test.shape[1]))
            res = y_train[i_neigh,:]-lr.predict(X_train[i_neigh,:])
            Q_xf[:, :, i] = np.cov(res.T)  
        mean_xf = y_llr.reshape((dx,N))
        for i in range(N):
            if len(ind_x) > 1:
                xf[:, i] = np.random.multivariate_normal(mean_xf[:, ind_x[i]],
                                                         Q_xf[:, :, ind_x[i]])
            else:
                xf[:, i] = np.random.multivariate_normal(mean_xf[:, i], Q_xf[:, :, i])

        M_xf = None
        return xf, mean_xf, Q_xf, M_xf
    
# parameters of the filtering method
class DA:
    
    def __init__(self,method='AnEnKS', N=20, xb, B, H, R, forecast_fct):
        method = 'AnEnKS' # chosen method ('AnEnKF', 'AnEnKS', 'AnPF')
        self.N = N # number of members (AnEnKF/AnEnKS) or particles (AnPF)
        self.xb = xb# xt.values[0,:]; B = 0.1*np.eye(xt.values.shape[1])
        self.H = H #H = np.eye(xt.values.shape[1])
        self.R = R #R = GD.sigma2_obs*np.eye(xt.values.shape[1])
    
    @staticmethod
    def m(x):
        return forecast_fct(x,AF)    
    
    def AnDA_data_assimilation(self,yo):
        """ Apply stochastic and sequential data assimilation technics using model forecasting or analog forecasting. """
    
        # dimensions
        n = len(self.xb)
        T = yo.values.shape[0]
        p = yo.values.shape[1] 
    
        # check dimensions
        if p!=self.R.shape[0]:
            print("Error: bad R matrix dimensions")
            quit()
    
        # initialization
        class x_hat:
            part = np.zeros([T,self.N,n]);
            weights = np.zeros([T,self.N]);
            values = np.zeros([T,n]);
            loglik = np.zeros([T]);
            time = yo.time;
    
        # EnKF and EnKS methods
        if (self.method =='AnEnKF' or self.method =='AnEnKS'):
            m_xa_part = np.zeros([T,self.N,n]);
            xf_part = np.zeros([T,self.N,n]);
            Pf = np.zeros([T,n,n]);
            for k in tqdm(range(0,T)):
                # update step (compute forecasts)            
                if k==0:
                    xf = np.random.multivariate_normal(self.xb, self.B, self.N);
                else:
                    xf, m_xa_part_tmp = self.m(x_hat.part[k-1,:,:]);
                    m_xa_part[k,:,:] = m_xa_part_tmp;         
                xf_part[k,:,:] = xf;
                Ef = np.dot(xf.T,np.eye(self.N)-np.ones([self.N,self.N])/self.N);
                Pf[k,:,:] = np.dot(Ef,Ef.T)/(self.N-1);
                # analysis step (correct forecasts with observations)          
                i_var_obs = np.where(~np.isnan(yo.values[k,:]))[0];            
                if (len(i_var_obs)>0):                
                    eps = np.random.multivariate_normal(np.zeros(len(i_var_obs)),self.R[np.ix_(i_var_obs,i_var_obs)],self.N);
                    yf = np.dot(self.H[i_var_obs,:],xf.T).T
                    SIGMA = np.dot(np.dot(self.H[i_var_obs,:],Pf[k,:,:]),self.H[i_var_obs,:].T)+self.R[np.ix_(i_var_obs,i_var_obs)]
                    SIGMA_INV = np.linalg.inv(SIGMA)
                    K = np.dot(np.dot(Pf[k,:,:],self.H[i_var_obs,:].T),SIGMA_INV)             
                    d = np.repeat(yo.values[k,i_var_obs][np.newaxis],self.N,0)+eps-yf
                    x_hat.part[k,:,:] = xf + np.dot(d,K.T)           
                    # compute likelihood
                    innov_ll = np.mean(np.repeat(yo.values[k,i_var_obs][np.newaxis],self.N,0)-yf,0)
                    loglik = -0.5*(np.dot(np.dot(innov_ll.T,SIGMA_INV),innov_ll))-0.5*(n*np.log(2*np.pi)+np.log(np.linalg.det(SIGMA)))
                else:
                    x_hat.part[k,:,:] = xf;          
                x_hat.weights[k,:] = np.repeat(1.0/self.N,self.N);
                x_hat.values[k,:] = np.sum(x_hat.part[k,:,:]*np.repeat(x_hat.weights[k,:][np.newaxis],n,0).T,0);
                x_hat.loglik[k] = loglik
            # end AnEnKF
            
            # EnKS method
            if (self.method == 'AnEnKS'):
                for k in tqdm(range(T-1,-1,-1)):           
                    if k==T-1:
                        x_hat.part[k,:,:] = x_hat.part[T-1,:,:];
                    else:
                        m_xa_part_tmp = m_xa_part[k+1,:,:];
                        tej, m_xa_tmp = self.m(np.mean(x_hat.part[k,:,:],0)[np.newaxis]);
                        tmp_1 =(x_hat.part[k,:,:]-np.repeat(np.mean(x_hat.part[k,:,:],0)[np.newaxis],self.N,0)).T;
                        tmp_2 = m_xa_part_tmp-np.repeat(m_xa_tmp,self.N,0);                    
                        Ks = 1.0/(self.N-1)*np.dot(np.dot(tmp_1,tmp_2),inv_using_SVD(Pf[k+1,:,:],0.9999));                    
                        x_hat.part[k,:,:] = x_hat.part[k,:,:]+np.dot(x_hat.part[k+1,:,:]-xf_part[k+1,:,:],Ks.T);
                    x_hat.values[k,:] = np.sum(x_hat.part[k,:,:]*np.repeat(x_hat.weights[k,:][np.newaxis],n,0).T,0);             
            # end AnEnKS  
        
        # particle filter method
        elif (self.method =='AnPF'):
            # special case for k=1
            k=0
            k_count = 0
            m_xa_traj = []
            weights_tmp = np.zeros(self.N);
            xf = np.random.multivariate_normal(self.xb, self.B, self.N)
            i_var_obs = np.where(~np.isnan(yo.values[k,:]))[0]
            if (len(i_var_obs)>0):
                # weights
                for i_N in range(0,self.N):
                    weights_tmp[i_N] = multivariate_normal.pdf(yo.values[k,i_var_obs].T,np.dot(self.H[i_var_obs,:],xf[i_N,:].T),self.R[np.ix_(i_var_obs,i_var_obs)]);
                # normalization
                weights_tmp = weights_tmp/np.sum(weights_tmp);
                # resampling
                indic = resampleMultinomial(weights_tmp);
                x_hat.part[k,:,:] = xf[indic,:];         
                weights_tmp_indic = weights_tmp[indic]/sum(weights_tmp[indic])
                x_hat.values[k,:] = sum(xf[indic,:]*(np.repeat(weights_tmp_indic[np.newaxis],n,0).T),0);
                # find number of iterations before new observation
                k_count_end = np.min(np.where(np.sum(1*~np.isnan(yo.values[k+1:,:]),1)>=1)[0]);
            else:
                # weights
                weights_tmp = np.repeat(1.0/N,N);
                # resampling
                indic = resampleMultinomial(weights_tmp);
            x_hat.weights[k,:] = weights_tmp_indic;
            
            for k in tqdm(range(1,T)):
                # update step (compute forecasts) and add small Gaussian noise
                xf, tej = self.m(x_hat.part[k-1,:,:]) +np.random.multivariate_normal(np.zeros(xf.shape[1]),self.B/100.0,xf.shape[0]);        
                if (k_count<len(m_xa_traj)):
                    m_xa_traj[k_count] = xf;
                else:
                    m_xa_traj.append(xf);
                k_count = k_count+1;
                # analysis step (correct forecasts with observations)
                i_var_obs = np.where(~np.isnan(yo.values[k,:]))[0];
                if len(i_var_obs)>0:
                    # weights
                    for i_N in range(0,self.N):
                        weights_tmp[i_N] = multivariate_normal.pdf(yo.values[k,i_var_obs].T,np.dot(self.H[i_var_obs,:],xf[i_N,:].T),self.R[np.ix_(i_var_obs,i_var_obs)]);
                    # normalization
                    weights_tmp = weights_tmp/np.sum(weights_tmp);
                    # resampling
                    indic = resampleMultinomial(weights_tmp);            
                    # stock results
                    x_hat.part[k-k_count_end:k+1,:,:] = np.asarray(m_xa_traj)[:,indic,:];
                    weights_tmp_indic = weights_tmp[indic]/np.sum(weights_tmp[indic]);            
                    x_hat.values[k-k_count_end:k+1,:] = np.sum(np.asarray(m_xa_traj)[:,indic,:]*np.tile(weights_tmp_indic[np.newaxis].T,(k_count_end+1,1,n)),1);
                    k_count = 0;
                    # find number of iterations  before new observation
                    try:
                        k_count_end = np.min(np.where(np.sum(1*~np.isnan(yo.values[k+1:,:]),1)>=1)[0]);
                    except ValueError:
                        pass
                else:
                    # stock results
                    x_hat.part[k,:,:] = xf;
                    x_hat.values[k,:] = np.sum(xf*np.repeat(weights_tmp_indic[np.newaxis],n,0).T,0);
                # stock weights
                x_hat.weights[k,:] = weights_tmp_indic;   
            # end AnPF
        
        # error
        else :
            print("Error: choose self.method between 'AnEnKF', 'AnEnKS', 'AnPF' ")
            quit()
        return x_hat       

#
#class LLRClass:
#    """
#    lagx:  lag of removed analogs around x to avoid an over-fitting forecast
#    lag_Dx:  15 lag of moving window of analogs chosen around x
#    time_period:  set 365.25 for year and
#    k_m: number of analogs
#    k_Q: number of analogs
#    nN_m:  number of analogs to be chosen for mean estimation
#    nN_Q: number of analogs to be chosen for dynamical error covariance
#    lag_k: chosen the lag for k reestimation
#    estK: set 'same' if k_m = k_Q chosen, otherwise, set 'different'
#    kernel: set 'rectangular' or 'tricube'
#    """
#
#    def __init__(self, data, data_prev, num_ana, Q):
#        self.data = data
#        self.data_prev = data_prev
#        self.lag_x = 5
#        self.lag_Dx = lambda Dx: np.shape(Dx)[-1]
#        self.time_period = 1
#        self.k_m = []
#        self.k_Q = []
#        self.nN_m = np.arange(20, num_ana, 50)
#        self.nN_Q = np.arange(20, num_ana, 50)
#        self.lag_k = 1
#        self.estK = 'same'
#        self.kernel = 'tricube'
#        self.k_lag = 20
#        self.k_inc = 10
#        self.Q = Q
#        self.gam = 1
#
#
#
#    def m_LLR(self, x, tx, ind_x):
#        # Local Linear Regression with kernel
#        # No cross validation -> fixed number of neighbors and bandwidth
#        # x: set of particles
#        # tx: current time associate to the set of particles
#        # ind_x: number of particles to return par input particle
#        
#        # initializations
#        dx, N = x.shape
#        xf = np.zeros([dx, N])
#        mean_xf = np.zeros([dx, N])
#        Q_xf = np.zeros([dx, dx, N])
#        M_xf = np.zeros([dx + 1, dx, N])
#        
#        T = self.data.shape[2]
#        ind_train = np.delete(np.arange(T),np.arange(max(tx-self.lag_x,0),min(tx+self.lag_x+1,T)))
#        #ind_train=np.arange(T)
#        N_train = self.data_prev.shape[1] # number of particles in the train dataset
#        X_train = self.data_prev[:,:,ind_train].reshape((dx,N_train*len(ind_train))).T
#        y_train = self.data[:,:,ind_train].reshape((dx,N_train*len(ind_train))).T
#        X_test = x.T
#        
#        if self.kernel=='tricube':
#            def wght(x,w_lambda):
#                return (1 - (x / w_lambda) ** 3) ** 3
#        else:
#            def wght(x,w_lambda):
#                return 1/len(x)    
#        neigh = NearestNeighbors(n_neighbors=n_neighbors) 
#        neigh.fit(X_train) 
#        Xn = neigh.kneighbors(X_test) # return_distance=True (default)
#        y_llr= np.zeros(X_test.shape)
#        lr = LinearRegression()
#        Q_xf=np.zeros((dx,dx,N))
#        for i in range(N):
#            i_neigh = Xn[1][i]
#            lr.fit(X_train[i_neigh,:],y_train[i_neigh,:],sample_weight=wght(Xn[0][i],w_lambda)) # add a weight!
#            y_llr[i,:] = lr.predict(X_test[i,:].reshape(1,X_test.shape[1]))
# #           if (self.Q.type == 'adaptive'):
# #               res = y_train[i_neigh,:]-lr.predict(X_train[i_neigh,:])
# #               Q_xf[:,:,i] = np.cov(res) # no kernel
# #           else:
#            Q_xf[:, :, i] = self.Q  
#        mean_xf = y_llr.reshape((dx,N))
#        for i in range(N):
#            if len(ind_x) > 1:
#                xf[:, i] = np.random.multivariate_normal(mean_xf[:, ind_x[i]],
#                                                         Q_xf[:, :, ind_x[i]])
#            else:
#                xf[:, i] = np.random.multivariate_normal(mean_xf[:, i], Q_xf[:, :, i])
#
#        M_xf = None
#        return xf, mean_xf, Q_xf, M_xf


