__author__ = "Pierre Navaro"
__email__ = "navaro@math.cnrs.fr"

import numpy as np

class Data:

    def __init__(self, dx, data_init, Y):

        ind_nogap = np.where(~np.isnan(np.sum(data_init,0)))[0] 
        self.ana  = np.zeros((dx, 1, len(ind_nogap)))
        self.suc  = np.zeros((dx, 1, len(ind_nogap)))
        self.ana[:, 0, :] = data_init[:dx, ind_nogap]
        self.suc[:, 0, :] = data_init[dx:, ind_nogap]
        self.time = Y.time[ind_nogap]
