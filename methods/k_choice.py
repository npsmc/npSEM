# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 13:21:13 2018

@author: trang
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LinearLocator, FormatStrFormatter

from methods.LLR_forecasting_CV import m_LLR


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

    def __init__(self, dx, data_init, ind_nogap, Y_train):
        self.ana = np.zeros((dx, 1, len(ind_nogap)))
        self.suc = np.zeros((dx, 1, len(ind_nogap)))
        self.ana[:, 0, :] = data_init[:dx, ind_nogap]
        self.suc[:, 0, :] = data_init[dx:, ind_nogap]
        self.time = Y_train.time[ind_nogap]


class LLRClass:
    """
    lagx:  lag of removed analogs around x to avoid an over-fitting forecast
    lag_Dx:  15 lag of moving window of analogs chosen around x
    time_period:  set 365.25 for year and
    k_m: number of analogs
    k_Q: number of analogs
    nN_m:  number of analogs to be chosen for mean estimation
    nN_Q: number of analogs to be chosen for dynamical error covariance
    lag_k: chosen the lag for k reestimation
    estK: set 'same' if k_m = k_Q chosen, otherwise, set 'different'
    kernel: set 'rectangular' or 'tricube'
    """
    def __init__(self, data, data_prev, num_ana, Q):
        self.data = data
        self.data_prev = data_prev
        self.lag_x = 5
        self.lag_Dx = lambda Dx: np.shape(Dx)[-1]
        self.time_period = 1
        self.k_m = []
        self.k_Q = []
        self.nN_m = np.arange(20, num_ana, 50)
        self.nN_Q = np.arange(20, num_ana, 50)
        self.lag_k = 1
        self.estK = 'same'
        self.kernel = 'tricube'
        self.k_lag = 20
        self.k_inc = 10
        self.Q = Q
        self.gam = 1

    def k_choice(self):
        x = self.data.ana
        y = self.data.suc
        time = self.data.time
        dx, N, T = x.shape
        if self.Q.type == 'fixed' or self.estK == 'same':

            print(" First case")

            L = []
            E = []

            for i in range(len(self.nN_m)):
                self.k_m = self.nN_m[i]
                self.k_Q = self.k_m
                loglik = 0
                err = 0
                for t in range(T):
                    _, mean_xf, Q_xf, _ = m_LLR(x[..., t], time[t], np.ones([1]), self)
                    innov = y[..., t] - mean_xf
                    err += np.mean(innov ** 2)

                L.append(loglik)
                E.append(np.sqrt(err / T))

            ind_max = np.argmin(E)
            k_m = self.nN_m[ind_max]
            k_Q = k_m
            plt.rcParams['figure.figsize'] = (8, 5)
            plt.figure(3)
            fig, ax1 = plt.subplots()
            ax1.plot(self.nN_m, E, color='b')
            ax1.set_xlabel('number of $m$-analogs $(k_m)$')
            ax1.set_ylabel('RMSE')
            plt.show()
        else:
            X, Y = np.meshgrid(self.nN_m, self.nN_Q)
            Q = np.zeros((dx, dx, T))
            len_ana = len(self.nN_m) * len(self.nN_Q)
            L = np.zeros(len_ana)
            E = np.zeros(len_ana)

            for i in range(len_ana):
                self.k_m = np.squeeze(X.T.reshape(len_ana)[i])
                if np.squeeze(Y.T.reshape(len_ana)[i]) > self.k_m:
                    indY = divmod(i, len(self.nN_Q))

                    # condition: number of analogs for m estimates is
                    # always larger or equal to the one for Q estimates
                    Y[indY[1], indY[0]] = self.k_m
                self.k_Q = np.squeeze(Y.T.reshape(len_ana)[i])
                loglik = 0
                err = 0
                for t in range(T):
                    _, mean_xf, Q_xf, _ = m_LLR(x[..., t], time[t], np.ones([1]), self)
                    innov = y[..., t] - mean_xf
                    const = -.5 * np.log(2 * np.pi * np.linalg.det(Q_xf.transpose(-1, 0, 1)))
                    logwei = -.5 * np.sum(
                        innov.T.dot(np.linalg.inv(Q_xf.transpose(-1, 0, 1)))[:N, :N, :] * innov.T, 1)
                    loglik += np.sum(const + logwei) / N
                    err += np.sqrt(np.mean(innov ** 2))

                L[i] = loglik
                E[i] = np.sqrt(err / T)
            ind_max = np.argmax(L)
            self.k_m = np.squeeze(X.T.reshape(len_ana)[ind_max])
            self.k_Q = np.squeeze(Y.T.reshape(len_ana)[ind_max])
            print(L)
            print(E)
            LL = (L.reshape((len(self.nN_m), len(self.nN_Q)))).T
            plt.rcParams['figure.figsize'] = (9, 9)
            fig = plt.figure(3)

            ax = fig.gca(projection='3d')
            surf = ax.plot_surface(X, Y, LL, cmap='Greys', alpha=0.75,
                                   linewidth=0.25, edgecolor='k', antialiased=False)
            #        plt.plot(k_m,k_Q,max(L),'k*', markersize = 6)
            ax.zaxis.set_major_locator(LinearLocator(10))
            ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
            fig.colorbar(surf, shrink=0.5, aspect=5)
            ax.set_xlabel('$k_m$')  # number of $m$-analogs
            ax.set_ylabel('$k_Q$')  # number of $Q$-analogs
            ax.set_zlabel('log likelihood')
            plt.grid()
            plt.show()

            print('Q ={}'.format(np.mean(Q, 2)))
        #        fig = plt.figure(4)
        #        plt.rcParams['figure.figsize'] = (5, 8)
        #        ax = fig.gca(projection='3d')
        #        surf = ax.plot_surface(X, Y, E.reshape((len(LLR.nN_m),len(LLR.nN_Q))), cmap='Blues',
        #                       linewidth=0, antialiased=False)
        #        ax.zaxis.set_major_locator(LinearLocator(10))
        #        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        #        fig.colorbar(surf, shrink=0.5, aspect=5)
        #        ax.set_xlabel('number of analogs $(k_m)$')
        #        ax.set_ylabel('number of analogs $(k_Q)$')
        #        ax.set_zlabel('RMSE')
        #        plt.grid()
        #        plt.show()

        #        fig = plt.figure(4)
        #        plt.rcParams['figure.figsize'] = (5, 8)
        # Add a color bar which maps values to colors.

