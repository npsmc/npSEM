import numpy as np
import matplotlib.pyplot as plt

from   npsem.models import SSM
import npsem.models.l63f as mdl_l63
from   npsem.models.l63 import l63_jac
 

sigma  = 10.0,
rho    = 28.0,
beta   = 8.0 / 3
h      = lambda x : H.dot(x)
jac_h  = lambda x : H
mx     = lambda x : fmdl.integrate(x)
jac_mx = lambda x, dt : l63_jac(x, dt, sigma, rho, beta)  

lorenz63 = SSM(h = h, jac_h = jac_h, mx = mx, jac_mx = jac_mx)
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

