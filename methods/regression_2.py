#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 11:09:01 2018

@author: trang
"""
import numpy as np
import matplotlib.pyplot as plt #plot
def regression_2(data,data0,data_perf,t, LLR):
  
    T,dx =np.shape(data);
    
    # #optimizing the value of k
    # tabk=100:100:1000;rms=zeros(size(tabk)); #
    # 
    # for j=1:length(tabk);
    #     k=tabk(j);#number of analogs
    # T=10^4;  #complete only the first dates to avoid lond computation (about 30 minutes to treat all the dataset)
    # #T=T  # to work on the full dataset
    # tic
    # for i=lag+1:T-lag,
    #     indY=find(isnan(data(i,:)));  #missing values
    #     indX=find(not(isnan(data(i,:)))); #observed components
    #     if length(indY)>0,  #if misisng values
    #         ind=find(abs(mod(t(i)-t,365.25))<15 & abs(t(i)-t)>2 & t > t(lag) & t < t(T-lag));  #dates in a 15 days window, but not in a 2 days time window
    #         X=[data(ind-1,:),data(ind,indX),data(ind+1,:)];  #covariate
    #         X=[ones(size(X,1),1),X];
    #         Y=data(ind,indY); #value to forecast
    #         Xn=[1,data(i-1,:),data(i,indX),data(i+1,:)];
    #         Yn=regloc_ana_miss(X,Y,Xn,k);
    #         datan(i,indY)=Yn;
    #     end
    # end
    # toc
    # err=data0(1:T,:)-datan(1:T,:);
    # ind=find(err(:).^2>0); #index of the observations which have been completed
    # ms(j)=mean(err(ind).^2)
    #end
    lag =1; lag_Dx = LLR.lag_Dx(data)
    datan = np.zeros((T,dx));#dataset with imputed values
    k=min(300,LLR.k_m);
    for i in range(lag,T-lag):
        indY= np.where(np.isnan(data[i,:]))[0];  #missing values
        indX= np.where(~ np.isnan(data[i,:]))[0]; #observed components
        if len(indY)>0:  #if misisng values
            ind= np.where(((np.abs(t[i]-t) % LLR.time_period)<lag_Dx) & (np.abs(t[i]-t)>LLR.lag_x) & (t > t[lag]) & (t < t[T-lag]))[0];  #dates in a 15 days window, but not in a 2 days time window
            X= np.concatenate((data[ind-1,:],data[np.ix_(ind,indX)],data[ind+1,:]),axis=1)  #covariate
            X= np.concatenate([np.ones((np.size(X,0),1)),X],axis=1);
            Y=data[np.ix_(ind,indY)]; #value to forecast
            Xn=np.concatenate(( np.reshape(1,(1,1)),np.reshape(data[i-1,:],(1,np.size(data[i-1,:]))),
                               np.reshape(data[i,indX],(1,np.size(data[i,indX]))),np.reshape(data[i+1,:],(1, np.size(data[i+1,:])))),axis =1);
            Yn=regloc_ana_miss(X,Y,Xn,k,LLR.kernel);
            datan[i,indY]=Yn; 
        datan[i,indX] = data[i,indX];
    datan[0,:] =data[0,:]; datan[T-lag,:] =data[T-lag,:];
    
    #save resreg.mat datan data
    

    err_y= data0-datan; err_x = data_perf-datan
    ind_gap= np.where(err_y**2>0)[0];  #permits to find dates with artificial missing values
    # plot(V(ind_gap),Vn(ind_gap),'.');  #scatterplot
    RMSE_y = np.sqrt(np.mean(err_y[ind_gap]**2));  #RMSE
    RMSE_x= np.sqrt(np.mean(err_x[ind_gap]**2));  #RMSE
    
    return datan, RMSE_y, RMSE_x, ind_gap
#%%
def regloc_ana_miss(X,Y,Xn,k,kernel):
    m=np.size(Xn,0);
    #keep only date where all the components of Y are observed
    indY= ~np.isnan(np.sum(Y,1)); 
    X=X[indY,:];
    Y=Y[indY,:];
    Yloc = np.zeros((m,np.size(Y,1)));
    for i in range(m):
        X2=np.squeeze(Xn[i,:]);
        #extract dates where the covariates are observed
        indX2= np.where(~ np.isnan(X2))[0];  #componennt observed
        X2=X2[indX2];
        Xr=X[:,indX2];
        ind= ~ np.isnan(np.sum(Xr,1)); 
        Xr=X[np.ix_(ind,indX2)];
        Yr=Y[ind,:];
        
        #regression lienaire locale
        dist= np.sum((Xr-np.tile(X2,(np.size(Xr,0),1)))**2,1);  # compute euclidean distance
        #distance
        ind_sort=np.argsort(dist); # sort with quick ort algorithm? 
        k_m = min(k,len(ind_sort))
        ind=ind_sort[np.arange(0,k_m)];
        #betai=(X'*(repmat(W,1,size(X,2)).*X))\(X'*(repmat(W,1,size(Y,2)).*Y));
        if kernel == 'tricube':
            h_m = dist[ind[-1]]; # chosen bandwidth to hold the constrain dist/h_m <= 1
            wei = (1-(dist[ind]/h_m)**3)**3;
        else:
            wei = np.ones(k_m);
        wei = wei/np.sum(wei); W=  np.sqrt(np.diag(wei));

        Aw = W.dot(Xr[ind,:]);
        Bw = W.dot(Yr[ind,:]); 	
        betai= np.linalg.lstsq(Aw,Bw)[0];    
        Yloc[i,:]=X2.dot(betai);
#        plt.plot(dist[ind]/h_m,wei)
#        plt.show()
    return Yloc