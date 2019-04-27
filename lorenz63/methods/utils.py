"""
Available on https://github.com/ptandeo/CEDA
@author: Pierre Tandeo

"""

import numpy as np

def sampling_discrete(W, m):  ### Discrete sampling, ADDED by TRANG ###
    "Returns m indices given N weights"
    cumprob = np.cumsum(W)
    n = np.size(cumprob)
    R = np.random.rand(m)
    ind = np.zeros(m)
    for i in range(n):
        ind += R> cumprob[i]
    ind = np.array(ind, dtype = int)    
    return ind


def resampling_sys(W): ### systematic resampling with respect to multinomial distribution, ADDED by TRANG ###
    "Returns a N-set of indices given N weights"
    N = np.size(W)
    u0 = np.random.rand(1)
    u = (range(N) +u0).T/N;
    qc = np.cumsum(W)
    qc = qc[:]
    qc = qc/qc[-1]
    new= np.concatenate((u,qc), axis=0)
    ind1 = np.argsort(new)
#   ind2 = np.where(ind1<=N-1);
    ind2 = np.array(np.where(ind1<=N-1),dtype = int)
    ind = ind2- range(N)
    a = ind[0,]
    return a
    
def inv_svd(A):
    "Returns the inverse matrix by SVD"
    U, s, V = np.linalg.svd(A, full_matrices=True)
    invs = 1./s
    n = np.size(s)
    invS = np.zeros((n,n))
    invS[:n, :n] = np.diag(invs)
    invA=np.dot(V.T,np.dot(invS, U.T))
    return invA

def sqrt_svd(A):
   "Returns the square root matrix by SVD"

   U, s, V = np.linalg.svd(A)#, full_matrices=True)

   sqrts = np.sqrt(s)
   n = np.size(s)
   sqrtS = np.zeros((n,n))
   sqrtS[:n, :n] = np.diag(sqrts)

   sqrtA=np.dot(V.T,np.dot(sqrtS, U.T))

   return sqrtA

def climat_background(f,Q,x0, T,prng): ### MODIF TRANG ###
  Nx = x0.size
  X = np.zeros((Nx, T))
  X[:,0] = x0
  for t in range(T-1):
    X[:,t+1] = f(X[:,t]) + sqrt_svd(Q).dot(prng.normal(size=Nx))
  xb = X.mean(1)
  B  = np.cov(X)
  x0_true = X[:,-1]
  return xb, B, x0_true


def gen_truth(f, x0, T, Q, prng):
  sqQ = sqrt_svd(Q)
  Nx = x0.size
  Xt = np.zeros((Nx, T+1))
  Xt[:,0] = x0
  for k in range(T):
    Xt[:,k+1] = f(Xt[:,k]) + sqQ.dot(prng.normal(size=Nx))
  return Xt

def gen_obs(h, Xt, R, nb_assim, prng):
  sqR = sqrt_svd(R)
  No = sqR.shape[0]
  T = Xt.shape[1] -1
  Yo = np.zeros((No, T))
  Yo[:] = np.nan
  for k in range(T):
    if k%nb_assim == 0:
      Yo[:,k] = h(Xt[:,k+1]) + sqR.dot(prng.normal(size=No))
  return Yo

def RMSE(E):
  return np.sqrt(np.mean(E**2))

def CV95(Xs,X):
    T = len(X)
    CIlowXs = np.percentile(Xs, 2.5, axis= 0)
    CIupXs = np.percentile(Xs, 97.5, axis= 0)
    cov_prob1= np.array(np.where(CIlowXs < X))
    cov_prob2= np.array(np.where(X< CIupXs))
    cov_prob= len(np.intersect1d(cov_prob1,cov_prob2))/T*100
    return cov_prob, CIlowXs, CIupXs