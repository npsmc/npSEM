import time as t 

import numpy as np
import matplotlib.pyplot as plt

import models.l63f as mdl_l63
from methods.CPF_BS_smoothing import _CPF
from methods.generate_data import generate_data
from methods.llr_forecasting_CV import LLRClass, Data, Est
from methods.model_forecasting import m_true
from models.L63 import l63_jac

begin = t.time()
"""
 GENERATE SIMULATED DATA (LORENZ-63 MODEL)
 
 dx                 : dimension of the state
 dt_int             : fixed integration time
 dt_model           : chosen number of model time step  \in [1, 25]
                      the larger dt_model the more nonlinear model
 var_obs            : indices of the observed variables
 dy                 : dimension of the observations
 H                  : first and third variables are observed
 h                  : observation model
 jacH               : Jacobian matrix  of the observation model(for EKS_EM only)
 sigma, rho, beta   : physical parameters
 Q_true             : model covariance
 R_true             : observation covariance
"""
dx = 3
dt_int = 0.01
dt_model = 8
var_obs = np.array([0, 1, 2])
dy = len(var_obs)
H = np.eye(dx)  # H = H[(0,2),:]
h = lambda x: H.dot(x)
jacH = lambda x: H
sigma = 10.0
rho = 28.0
beta = 8.0 / 3
fmdl = mdl_l63.M(sigma=sigma, rho=rho, beta=beta, dtcy=dt_int)
mx = lambda x: fmdl.integ(x)  # fortran version (fast)
jac_mx = lambda x: l63_jac(x, dt_int * dt_model, sigma, rho, beta)  # python version (slow)

# Setting covariances
sig2_Q = 1;
sig2_R = 2  # parameters
Q_true = np.eye(dx) * sig2_Q
R_true = np.eye(dx) * sig2_R

# prior state
x0 = np.r_[8, 0, 30]

# generate data
T_burnin = 5 * 10 ** 3
T_train = 10 * 10 ** 2  # length of the catalog
T_test = 10 * 10 ** 2  # length of the testing data
X_train, Y_train, X_test, Y_test, yo = generate_data(x0, mx, h, Q_true,
                                                     R_true, dt_int, dt_model,
                                                     var_obs, T_burnin,
                                                     T_train, T_test)
X_train.time = np.arange(0, T_train)
Y_train.time = X_train.time[1:]
# np.random.seed(0);# random number generator
N = np.size(Y_train.values)
Ngap = N // 10  # create gaps: 10 percent of missing values
indX = np.random.choice(np.arange(0, N), Ngap, replace=False)
ind_gap_taken = divmod(indX, len(Y_train.time))
Y_train.values[ind_gap_taken] = np.nan

X_train0, Y_train0, X_test0, Y_test0, yo0 = generate_data(X_train.values[:, -1],
                                                          mx, h, Q_true, R_true,
                                                          dt_int,
                                                          dt_model, var_obs, T_burnin,
                                                          T_train, T_test)
X_train0.time = np.arange(0, T_train)
Y_train0.time = X_train0.time[1:]

# %% TEST FOR LOCAL LINEAR REGRESSION (LLR_forecasting_CV function)
data_init = np.r_['0,2,0', Y_train.values[..., :-1], Y_train.values[..., 1:]]
ind_nogap = np.where(~np.isnan(np.sum(data_init, 0)))[0]
num_ana = 200

estQ =  Est(value=Q_true,type='adaptive',form='constant',base=np.eye(dx),decision=True)
estR =  Est(value=R_true,type='adaptive',form='constant',base=np.eye(dx),decision=True)
estX0 = Est(decision = False)
estD =  Est(decision = True)

data = Data(dx, data_init, ind_nogap, Y_train)
data_prev = Data(dx, data_init, ind_nogap, Y_train)

data_prev.ana = data_init[:dx, ind_nogap]
data_prev.suc = data_init[dx:, ind_nogap]
data_prev.time = Y_train.time[ind_nogap]

LLR = LLRClass(data, data_prev, num_ana, estQ)

LLR.k_choice()

plt.rcParams['figure.figsize'] = (8, 5)
plt.figure(3)
fig, ax1 = plt.subplots()
ax1.plot(LLR.nN_m, LLR.E, color='b')
ax1.set_xlabel('number of $m$-analogs $(k_m)$')
ax1.set_ylabel('RMSE')
plt.show()


# Forecast with LLR
xf, mean_xf, Q_xf, M_xf = LLR.m_LLR(X_test.values[:,:-1], 1, np.ones([1]))

# %% TEST FOR CONDITIONAL PARTICLE FILTERING (_CPF function)
# Here I use the true model  (m) to forecast instead of LLR (m_hat).

m = lambda x, pos_x, ind_x: m_true(x, pos_x, ind_x, Q_true, 
                                   mx, jac_mx, dt_model)
# m_hat = lambda  x,pos_x,ind_x: m_LLR(x,pos_x,ind_x,LLR)

time = np.arange(T_test - 1)  # [np.newaxis].T
B = Q_true
xb = X_test.values[..., 0]

Nf = 10  # number of particles
Ns = 5  # number of realizations

X_conditioning = np.zeros([dx, T_test])

Xa, Xf, mean_Xf, cov_Xf, Wa, ind, loglik = _CPF(Y_test.values,
                                                X_conditioning,
                                                m, H, Q_true, R_true,
                                                xb, B, Nf, dx,
                                                time) 
print(f"Elapsed time : {t.time()-begin}")
