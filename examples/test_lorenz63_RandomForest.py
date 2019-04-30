#!/usr/bin/env python
# coding: utf-8

# In[187]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")


# In[188]:


import numpy as np 
from numpy.linalg import cholesky 
import matplotlib.pyplot as plt 
from tqdm import tqdm_notebook as tqdm 


# In[207]:


import models.l63f as mdl_l63
from models.L63 import l63_jac
from methods.generate_data import generate_data
from methods.llr_forecasting_CV import LLRClass, Data, Est
from methods.model_forecasting import m_true
from models.L63 import l63_jac
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LinearRegression

#%%
def compare_LLR_GB(X_train, Y_train, X_test, Y_test):

    # Gradient boosting
    max_depth = 3
    n_estimators = 200
    max_features = 2
    learning_rate = .025
    clf = MultiOutputRegressor(GradientBoostingRegressor(
                                n_estimators=n_estimators,
                                learning_rate=learning_rate,
                                max_depth=max_depth,
                                max_features = max_features,
                                #min_samples_split = min_samples_split,
                                #min_samples_leaf = min_samples_leaf,
                                random_state=0))
    clf.fit(X_train, y_train)
    # Predict on new data
    y_gb = clf.predict(X_test)
    # LLR
    neigh = NearestNeighbors(n_neighbors=70)
    neigh.fit(X_train) 

    Xn = neigh.kneighbors(X_test)
    y_llr= np.zeros(y_test.shape)
    lr = LinearRegression()
    for i in range(X_test.shape[0]):
        i_neigh = Xn[1][i]
        lr.fit(X_train[i_neigh,:],y_train[i_neigh,:]) # add a weight!
        y_llr[i,:] = lr.predict(X_test[i,:].reshape(1,3))
        
    rmse_gb = np.sqrt(np.mean((y_test-y_gb)**2))
    rmse_llr = np.sqrt(np.mean((y_test-y_llr)**2))
    return rmse_gb, rmse_llr

   

#%% GENERATE SIMULATED DATA (LORENZ-63 MODEL)
dx = 3 # dimension of the state
dt_int = 0.01 # fixed integration time
dt_model = 8 # chosen number of model time step  \in [1, 25]-the larger dt_model the more nonliner model
var_obs = np.array([0,1,2]) # indices of the observed variables
dy = len(var_obs) # dimension of the observations
H = np.eye(dx)
#H = H[(0,2),:] #  first and third variables are observed
h = lambda x: H.dot(x)  # observation model
jacH = lambda x: H # Jacobian matrix  of the observation model(for EKS_EM only)
sigma = 10.0
rho = 28.0
beta = 8.0/3 # physical parameters
fmdl = mdl_l63.M(sigma=sigma, rho=rho, beta=beta, dtcy= dt_int)
mx = lambda x: fmdl.integ(x) # fortran version (fast)
jac_mx = lambda x: l63_jac(x, dt_int*dt_model, sigma, rho, beta) # python version (slow)

# Setting covariances
sig2_Q = 1; sig2_R = 2 # parameters
Q_true = np.eye(dx) *sig2_Q # model covariance
R_true = np.eye(dx) *sig2_R # observation covariance

# prior state
x0 = np.r_[8, 0, 30]

# generate data
T_burnin = 5*10**3
T_train = 10*10**2 # length of the catalog
T_test = 10*10**2 # length of the testing data


B = 10
rmse_gb, rmse_llr = [],[]
for b in range(B):
    print(b)
    X_train_ref, Y_train_ref, X_test_ref, Y_test_ref, yo = generate_data(x0,
                                                     mx,h,Q_true,R_true,
                                                     dt_int,dt_model,var_obs, 
                                                     T_burnin, T_train, T_test,seed=b) 
    X_train, X_test = X_train_ref.values[:,:-1].T, X_test_ref.values[:,:-1].T
    y_train, y_test = X_train_ref.values[:,1:].T, X_test_ref.values[:,1:].T
    gb_sc, llr_sc  = compare_LLR_GB(X_train, y_train, X_test, y_test)
    rmse_gb.append(gb_sc)
    rmse_llr.append(llr_sc)

print("GB mean score:  ", np.round(np.mean(rmse_gb),2), "(" , np.round(np.std(rmse_gb),2), ")" )
print("LLR mean score:  ", np.round(np.mean(rmse_llr),2), "(" , np.round(np.std(rmse_llr),2), ")" )

#%% Examples 

plt.plot(X_train.time,X_train.values[0,:])
plt.title("Time series, 1st component")

X = X_train.values[:,:-1].T
y = X_train.values[:,1:].T

print(X.shape)
print(y.shape)
print(X_test.values.shape)

plt.figure()
plt.plot(X[:,0],y[:,0],'.')
plt.title("Scatter plot, dt=1, 1st component")

# In[193]:
# # Random Forest


X_train, X_test  = X[:800,:], X[800:,]
y_train, y_test  = y[:800,:], y[800:,]



max_depth = 3
n_estimators = 500
regr_multirf = MultiOutputRegressor(RandomForestRegressor(n_estimators=n_estimators,
                                                          max_depth=max_depth,
                                                          random_state=0))
regr_multirf.fit(X_train, y_train)

regr_rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                random_state=2) 
regr_rf.fit(X_train, y_train)

# Predict on new data
y_multirf = regr_multirf.predict(X_test)
y_rf = regr_rf.predict(X_test)


# In[194]:


# rmse scores
print("Multi RF score=%.2f" % np.sqrt(np.mean((y_test-y_multirf)**2)))
print("RF score=%.2f" % np.sqrt(np.mean((y_test-y_rf)**2)))

plt.figure(figsize=[20,8])
for j in range(3):
    plt.subplot(1,3,j+1)
    plt.plot(y_test[:,j])
    plt.plot(y_multirf[:, j])
    plt.plot(y_rf[:, j])



# In[195]:
# # Gradient Boosting

parameters = {
    "learning_rate": [0.025, 0.05, 0.1,0.2],
    "max_depth":[3,5,8],
    "max_features":[1,2,3],
    "n_estimators":[10,50,100,200,500,1000]
    }

clf = GridSearchCV(GradientBoostingRegressor(), 
                            parameters, cv=10, n_jobs=-1,verbose=1)

clf.fit(X_train, y_train[:,1])
print(clf.best_params_)


# In[204]:


max_depth = 3
n_estimators = 200
max_features = 2
learning_rate = .025
clf = MultiOutputRegressor(GradientBoostingRegressor(
                                n_estimators=n_estimators,
                                learning_rate=learning_rate,
                                max_depth=max_depth,
                                max_features = max_features,
                                #min_samples_split = min_samples_split,
                                #min_samples_leaf = min_samples_leaf,
                                random_state=0))
clf.fit(X_train, y_train)

# Predict on new data
y_gb = clf.predict(X_test)


# In[205]:


print("Multi RF score=%.2f" % np.sqrt(np.mean((y_test-y_multirf)**2)))
print("RF score=%.2f" % np.sqrt(np.mean((y_test-y_rf)**2)))
print("GB score=%.2f" % np.sqrt(np.mean((y_test-y_gb)**2)))

plt.figure(figsize=[20,8])
for j in range(3):
    plt.subplot(1,3,j+1)
    plt.plot(y_test[:,j])
    plt.plot(y_gb[:, j])    
    plt.plot(y_rf[:, j])


# # LLR

# In[198]:



neigh = NearestNeighbors(n_neighbors=70)
neigh.fit(X_train) 

Xn = neigh.kneighbors(X_test)
y_llr= np.zeros(y_test.shape)
lr = LinearRegression()
for i in range(X_test.shape[0]):
    i_neigh = Xn[1][i]
    lr.fit(X_train[i_neigh,:],y_train[i_neigh,:]) # add a weight!
    y_llr[i,:] = lr.predict(X_test[i,:].reshape(1,3))
    #y_llr = np.concatenate((y_llr,lr.predict(X_test[i,:].reshape((1,3)))))


# In[199]:


# rmse scores
print("Multi RF score=%.2f" % np.sqrt(np.mean((y_test-y_multirf)**2)))
print("RF score=%.2f" % np.sqrt(np.mean((y_test-y_rf)**2)))
print("GB score=%.2f" % np.sqrt(np.mean((y_test-y_gb)**2)))
print("LLR score=%.2f" % np.sqrt(np.mean((y_test-y_llr)**2)))

plt.figure(figsize=[20,16])
for j in range(3):
    plt.subplot(2,3,j+1)
    plt.plot(y_test[:,j],'k',linewidth=2)
    plt.plot(y_llr[:, j])    
    plt.plot(y_gb[:, j])
for j in range(3):
    plt.subplot(2,3,j+4)
    plt.plot(y_test[:,j],y_llr[:, j],'.',label="LLR")
    plt.plot(y_test[:,j],y_gb[:, j],'.',label="GB")
plt.legend()


