#!/usr/bin/env python
# coding: utf-8

# In[187]:


#%% Modules importation

import numpy as np 
from   numpy.linalg import cholesky 
import matplotlib.pyplot as plt 
from   tqdm import tqdm_notebook as tqdm 

import npsem.models.l63f as mdl_l63
from   npsem.models.l63 import l63_jac
import npsem.models.l96f as mdl_l96
from   npsem.methods.generate_data import generate_data
from   npsem.methods.model_forecasting import m_true

from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor

from sklearn import preprocessing

import time

#%%
def LLR(X_train,X_test,n_neighbors=300,w_lambda=1): 
    neigh = NearestNeighbors(n_neighbors=n_neighbors) 
    neigh.fit(X_train) 
    Xn = neigh.kneighbors(X_test) # return_distance=True (default)
    y_llr= np.zeros(X_test.shape)
    lr = LinearRegression()
    for i in range(X_test.shape[0]):
        i_neigh = Xn[1][i]
        lr.fit(X_train[i_neigh,:],y_train[i_neigh,:],sample_weight=np.exp(-Xn[0][i]/w_lambda)) # add a weight!
        y_llr[i,:] = lr.predict(X_test[i,:].reshape(1,nx))
    return y_llr

def LLR_Ridge(X_train,X_test,n_neighbors=300,w_lambda=1): 
    neigh = NearestNeighbors(n_neighbors=n_neighbors) 
    neigh.fit(X_train) 
    Xn = neigh.kneighbors(X_test) # return_distance=True (default)
    y_llr= np.zeros(X_test.shape)
    lr = Ridge(alpha=.1)
    for i in range(X_test.shape[0]):
        i_neigh = Xn[1][i]
        lr.fit(X_train[i_neigh,:],y_train[i_neigh,:],sample_weight=np.exp(-Xn[0][i]/w_lambda)) # add a weight!
        y_llr[i,:] = lr.predict(X_test[i,:].reshape(1,nx))
    return y_llr

def compare_LLR_ANN(X_train, Y_train, X_test, Y_test,llr_params_,ann_params_):
    scaler = preprocessing.StandardScaler().fit(X_train)
    scaler.transform(X_train)
    scaler.transform(X_test)
    nx = X_train.shape[1]
    # Gradient boosting
    hidden_layer_sizes = ann_params_["hidden_layer_sizes"]
    alpha = ann_params_["alpha"]
    start = time.time()
    clf = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,
                                            alpha=alpha,
                                            learning_rate="adaptive",
                                            max_iter=5000)
    clf.fit(X_train, y_train)
    # Predict on new data
    y_ann = clf.predict(X_test)
    end = time.time()
    ann_time = end - start
    # LLR
    n_neighbors = llr_params_["n_neighbors"]
    w_lambda = llr_params_["w_lambda"]
    start = time.time()
    y_llr = LLR(X_train,X_test,n_neighbors=n_neighbors,w_lambda=w_lambda)
    end = time.time()
    llr_time = end - start
    start = time.time()
    y_llr_ridge = LLR_Ridge(X_train,X_test,n_neighbors=n_neighbors,w_lambda=w_lambda)
    end = time.time()
    ridge_time = end - start
    rmse_ann = np.sqrt(np.mean((y_test-y_ann)**2))
    rmse_llr = np.sqrt(np.mean((y_test-y_llr)**2))
    rmse_llr_ridge = np.sqrt(np.mean((y_test-y_llr_ridge)**2))
    return rmse_ann, rmse_llr, rmse_llr_ridge, ann_time, llr_time, ridge_time, y_ann, y_llr, y_llr_ridge

   
#%%
plt.plot(y_test[1:200,1],y_llr[1:200,1],'.')
plt.plot(y_test[1:200,1],y_gb[1:200,1],'.')
plt.plot([-10,15],[-10,15])
#%% GENERATE SIMULATED DATA (LORENZ-96 MODEL)
nx = 10 # dimension of the state
dt_int = 0.01 # fixed integration time
dt_model = 8
var_obs = np.array([0,1,2]) # indices of the observed variables
dy = len(var_obs) # dimension of the observations

sig2_Q = 1; sig2_R = 2 # parameters

h = lambda x: H.dot(x)  # observation model
jacH = lambda x: H # Jacobian matrix  of the observation model(for EKS_EM only)


B = 10
grid_search=False

# generate data
T_burnin = 5*10**3
T_train = 10*10**2 # length of the catalog
T_test = 10*10**2 # length of the testing data

RMSE_GB = []
RMSE_LLR = []
gb_tps, llr_tps = [], []
for nx in [10,15,20,25,30,35,40,45,50]: 
    print("nx = ",nx)
    H = np.eye(nx)
    #H = H[(0,2),:] #  first and third variables are observed
    force = 8.
    fmdl = mdl_l96.M(dtcy=0.05,force=force,nx=nx)
    mx = lambda x: fmdl.integ(x) # fortran version (fast)
    # Setting covariances
    Q_true = np.eye(nx) *sig2_Q # model covariance
    R_true = np.eye(nx) *sig2_R # observation covariance
    # prior state
    x0 = np.ones(nx)*force+np.random.normal(0,1,nx)
    
    ### Here : try to find the best parameters
    ### Ajouter un boolean optim = yes/no
    if grid_search:
        X_train_ref, Y_train_ref, X_test_ref, Y_test_ref, yo = generate_data(x0,
                                                         mx,h,Q_true,R_true,
                                                         dt_int,dt_model,var_obs, 
                                                         T_burnin, T_train, T_test,seed=b) 
 #       X_train, X_test = X_train_ref.values[:,:-1].T, X_test_ref.values[:,:-1].T
 #       y_train, y_test = X_train_ref.values[:,1:].T, X_test_ref.values[:,1:].T
        X_train = np.concatenate((X_train_ref.values[:,:-2].T,X_train_ref.values[:,1:-1].T))
        X_test = np.concatenate((X_test_ref.values[:,:-2].T,X_test_ref.values[:,1:-1].T))
        y_train, y_test = X_train_ref.values[:,2:].T, X_test_ref.values[:,2:].T
        # Gradient boosting
        parameters = {
        "learning_rate": [0.001,0.025, 0.05, 0.1],
        "max_depth":[3,5,8,12],
        "max_features":[2,5,10],
        "n_estimators":[50,100,200,500]
        }
        clf = GridSearchCV(GradientBoostingRegressor(), 
                                parameters, cv=5, n_jobs=-1,verbose=1)
        clf.fit(X_train, y_train[:,1])
        gb_best_params_ = clf.best_params_
        print("gb_best_params_ = ", gb_best_params_)
        # LLR
        J = 5
        n_neighbors=[50,100,150,200,300,500]
        sc = np.zeros((J,len(n_neighbors)))
        for i,n in enumerate(n_neighbors):
            for j in range(J):
                X_tr, X_te, y_tr, y_te = train_test_split(X_train,y_train,test_size=.2)
                y_res = LLR(X_tr,X_te,n_neighbors=n)
                sc[j,i] = np.sqrt(np.mean((y_te-y_res)**2))
        n_best = n_neighbors[np.argmin(np.mean(sc,axis=0))]     
        print("n_best = ", n_best)
    else:
        llr_best_params_ = {"n_neighbors": 500,"w_lambda": 20}
        gb_best_params_ = {"learning_rate": 0.05, "max_depth": 8, "max_features": 5, "n_estimators": 200} 
        
    rmse_gb, rmse_llr = [],[]
    for b in range(B):
        print(str(b)+" ", end='')
        X_train_ref, Y_train_ref, X_test_ref, Y_test_ref, yo = generate_data(x0,
                                                         mx,h,Q_true,R_true,
                                                         dt_int,dt_model,var_obs, 
                                                         T_burnin, T_train, T_test,seed=b) 
        X_train, X_test = X_train_ref.values[:,:-1].T, X_test_ref.values[:,:-1].T
        y_train, y_test = X_train_ref.values[:,1:].T, X_test_ref.values[:,1:].T
        gb_sc, llr_sc, gb_time, llr_time  = compare_LLR_GB(X_train, y_train, X_test, y_test,
                                                           llr_params_=llr_best_params_,gb_params_=gb_best_params_)
        rmse_gb.append(gb_sc)
        rmse_llr.append(llr_sc)
        gb_tps.append(gb_time)
        llr_tps.append(llr_time)

    RMSE_GB.append(rmse_gb)
    RMSE_LLR.append(rmse_llr)
    print(" ")
    print("GB mean score:  ", np.round(np.mean(rmse_gb),2), "(" , np.round(np.std(rmse_gb),2), ")" )
    print("LLR mean score:  ", np.round(np.mean(rmse_llr),2), "(" , np.round(np.std(rmse_llr),2), ")" )

#%% Forecasting errors
x = [10,15,20,25,30,35,40,45,50]
y_gb, y_gb_sd, y_llr, y_llr_sd = np.zeros(len(x)), np.zeros(len(x)), np.zeros(len(x)), np.zeros(len(x))
for i in range(len(x)):
    y_gb[i] = np.mean(RMSE_GB[i])
    y_gb_sd[i] = np.std(RMSE_GB[i])
    y_llr[i] = np.mean(RMSE_LLR[i])
    y_llr_sd[i] = np.std(RMSE_LLR[i])

plt.figure()
plt.plot(x,y_gb,label="GB")
plt.fill_between(x, y_gb-y_gb_sd, y_gb+y_gb_sd,alpha=.5)
plt.plot(x,y_llr,label="LLR")
plt.fill_between(x, y_llr-y_llr_sd, y_llr+y_llr_sd,alpha=.5)
plt.xlabel("Number of components")
plt.ylabel("forecast rmse")
plt.title("Lorenz 96")
plt.grid()
plt.legend()
#plt.savefig("L96_forecasting_error_chosen_params.png",format="png")

#%% Computation time
plt.figure()
plt.plot(x,np.mean(np.array(gb_tps).reshape((10,len(x))),axis=0),label="GB")
plt.plot(x,np.mean(np.array(llr_tps).reshape((10,len(x))),axis=0),label="LLR")
plt.xlabel("Number of components")
plt.ylabel("computation time (s)")
plt.title("Lorenz 96")
plt.grid()
plt.legend()
#plt.savefig("L96_computation_time_chosen_params.png",format="png")



