#!/usr/bin/python

import numpy as np
import npsem.models.l63_for as tfor

class M:

    """
    Lorenz-63 model. Wrapper to fortran L63 code
    """

    def __init__(self,sigma=10,rho=28,beta=8./3,dtcy=0.01):
        "Lorenz-63 parameters"
        self.sigma=sigma
        self.rho=rho
        self.beta=beta
        self.kt=10
        self.dtcy=dtcy
        self.dt=dtcy/self.kt # integration time step
        self.nx=3

    def integrate(self,xold):
        "Time integration of Lorenz-63 (single and ensemble)"

        par=np.zeros(self.nx)
        par[0]=self.sigma
        par[1]=self.rho
        par[2]=self.beta
    
        x=np.zeros(self.nx)

        if xold.ndim==1:
            #single integration
            x=tfor.tinteg_l63(xold,par,self.kt,self.dt) 
        else:
            #ensemble integration
            x=tfor.tintegem_l63(xold,par,self.kt,self.dt)
        return x
