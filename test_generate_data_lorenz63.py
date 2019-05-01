import numpy as np
import matplotlib.pyplot as plt

from   npsem.models import SSM
import npsem.models.l63f as mdl_l63
from   npsem.models.l63 import l63_jac
 

dt_int   = 0.01
dt_model = 8
var_obs  = [0, 1, 2]
sig2_Q   = 1
sig2_R   = 2 
x0       = [8, 0, 30]

dx       = len(x0)
H        = np.eye(dx)
h        = lambda x : H.dot(x)
jac_h    = lambda x : H

sigma, rho, beta    = 10.0, 28.0, 8.0 / 3
fmdl     = mdl_l63.M(sigma, rho, beta, dt_int)
mx       = lambda x : fmdl.integrate(x)
jac_mx   = lambda x, dt : l63_jac(x, dt, sigma, rho, beta)  

lorenz63 = SSM(h, jac_h, mx, jac_mx, dt_int, dt_model, x0, var_obs, 
               sig2_Q, sig2_R)

T_burnin = 5  * 10**3
T        = 20 * 10**2
X, Y     = lorenz63.generate_data( T_burnin, T)

fig, axes = plt.subplots(figsize = (15, 5))
axes.plot(X.values[:,1:].T,'-', color='grey')
axes.plot(Y.values.T,'.k', markersize= 6)
axes.set_xlabel('Lorenz-63 times')
axes.set_title('Lorenz-63 true (continuous lines) '
               'and observed trajectories (points)')
plt.savefig("lorenz63_data.png")

