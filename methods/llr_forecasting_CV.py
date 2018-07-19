# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 13:21:13 2018

@author: trang
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LinearLocator, FormatStrFormatter


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

            self.L, self.E = [], []

            for i in range(len(self.nN_m)):
                self.k_m = self.nN_m[i]
                self.k_Q = self.k_m
                loglik = 0
                err = 0
                for t in range(T):
                    _, mean_xf, Q_xf, _ = self.m_LLR(
                        x[:,:, t], time[t], np.ones([1]))
                    innov = y[:,:, t] - mean_xf
                    err += np.mean(innov ** 2)

                self.L.append(loglik)
                self.E.append(np.sqrt(err / T))

            ind_max = np.argmin(self.E)
            k_m = self.nN_m[ind_max]
            k_Q = k_m

        else:
            X, Y = np.meshgrid(self.nN_m, self.nN_Q)
            Q = np.zeros((dx, dx, T))
            len_ana = len(self.nN_m) * len(self.nN_Q)
            self.L = np.zeros(len_ana)
            self.E = np.zeros(len_ana)

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
                    _, mean_xf, Q_xf, _ = self.m_LLR(
                        x[:,:, t], time[t], np.ones([1]))
                    innov = y[:,:, t] - mean_xf
                    const = -.5 * \
                        np.log(2 * np.pi * np.linalg.det(Q_xf.transpose(-1, 0, 1)))
                    logwei = -.5 * np.sum(
                        innov.T.dot(np.linalg.inv(Q_xf.transpose(-1, 0, 1)))[:N, :N, :] * innov.T, 1)
                    loglik += np.sum(const + logwei) / N
                    err += np.sqrt(np.mean(innov ** 2))

                self.L[i] = loglik
                self.E[i] = np.sqrt(err / T)
            ind_max = np.argmax(self.L)
            self.k_m = np.squeeze(X.T.reshape(len_ana)[ind_max])
            self.k_Q = np.squeeze(Y.T.reshape(len_ana)[ind_max])
            print(" \t L = {} ".format(self.L))
            print(" \t E = {} ".format(self.E))

            print(" \t Q = {}".format(np.mean(Q, 2)))
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

            LL = (self.L.reshape((len(self.nN_m), len(self.nN_Q)))).T
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

    def m_LLR(self, x, tx, ind_x):
        """ Apply the analog method on data of historical data to generate forecasts. """

        # initializations
        dx, N = x.shape
        xf = np.zeros([dx, N])
        mean_xf = np.zeros([dx, N])
        Q_xf = np.zeros([dx, dx, N])
        M_xf = np.zeros([dx + 1, dx, N])

        lag_x = self.lag_x
        lag_Dx = self.lag_Dx(self.data.ana)
        if len(self.data.ana.shape) == 1:
            dimx = 1
            dimD = 1  # lenC = LLR.data.ana.shape
        elif len(self.data.ana.shape) == 2:
            dimD = 1
            dimx = self.data.ana.shape[0]
        else:
            dimx, dimD = self.data.ana.shape[:2]
        try:
            indCV = (np.abs((tx - self.data.time)) % self.time_period <= lag_Dx) \
                & (np.abs(tx - self.data.time) >= lag_x)
        except:
            indCV = (np.abs(tx - self.data.time) >= lag_x)

        lenD = np.shape(self.data.ana[..., np.squeeze(indCV)])[-1]
        analogs_CV = np.reshape(
            self.data.ana[:,:, np.squeeze(indCV)], (dimx, lenD * dimD))
        successors_CV = np.reshape(
            self.data.suc[:,:, np.squeeze(indCV)], (dimx, lenD * dimD))

        if self.gam != 1:
            if len(self.data_prev.ana.shape) == 1:
                dimx_prev = 1
                dimD_prev = 1  # lenC = LLR.data.ana.shape
            elif len(self.data_prev.ana.shape) == 2:
                dimD_prev = 1
                dimx_prev, _ = self.data_prev.ana.shape
            else:
                dimx_prev, dimD_prev, _ = self.data_prev.ana.shape
            indCV_prev = (indCV) & (self.data_prev.time == self.data_prev.time)
            lenD_prev = np.shape(self.data_prev.ana[:,:, indCV_prev])[-1]

            analogs_CV_prev = np.reshape(self.data_prev.ana[:,:, indCV_prev],
                                         (dimx_prev, lenD_prev * dimD_prev))
            successors_CV_prev = np.reshape(self.data_prev.suc[:,:, indCV_prev],
                                            (dimx_prev, lenD_prev * dimD_prev))
            analogs = np.concatenate((analogs_CV, analogs_CV_prev), axis=1)
            successors = np.concatenate(
                (successors_CV, successors_CV_prev), axis=1)
        else:
            analogs = analogs_CV
            successors = successors_CV

        # LLR.k_m = np.size(analogs,1)/np.size(analogs_CV,1)*LLR.LLR.k_m;
        # %% LLR estimating
        # #[ind_knn,dist_knn]=knnsearch(analogs,x,'k',LLR.k_m);
        self.k_m = min(self.k_m, np.size(analogs, 1))
        self.k_Q = min(self.k_m, self.k_Q)
        # rectangular kernel for default
        weights = np.ones((N, self.k_m)) / self.k_m

        for i in range(N):
            # search k-nearest neighbors
            X_i = np.tile(x[:, i], (np.size(analogs, 1), 1)).T
            dist = np.sqrt(np.sum((X_i - analogs) ** 2, 0))
            ind_dist = np.argsort(dist)
            ind_knn = ind_dist[:self.k_m]

            if self.kernel == 'tricube':
                # chosen bandwidth to hold the constrain dist/h_m <= 1
                h_m = dist[ind_knn[-1]]
                weights[i, :] = (1 - (dist[ind_knn] / h_m) ** 3) ** 3
            # identify which set analogs belong to (if using SAEM)
            ind_prev = np.where(ind_knn > np.size(analogs_CV, 1))
            ind = np.setdiff1d(np.arange(0, self.k_m), ind_prev)
            if len(ind_prev) > 0:
                weights[i, ind_prev] = (1 - self.gam) * weights[i, ind_prev]
                weights[i, ind] = self.gam * weights[i, ind]

            wei = weights[i, :] / np.sum(weights[i, :])
            # LLR coefficients
            W = np.sqrt(np.diag(wei))
            Aw = np.dot(np.insert(analogs[:, ind_knn], 0, 1, 0), W)
            Bw = np.dot(successors[:, ind_knn], W)
            M = np.linalg.lstsq(Aw.T, Bw.T)[0]
            # weighted mean and covariance
            mean_xf[:, i] = np.dot(np.insert(x[:, i], 0, 1, 0), M)
            M_xf[:, :, i] = M

            if (self.Q.type == 'adaptive'):
                res = successors[:, ind_knn] \
                    - np.dot(np.insert(analogs[:, ind_knn], 0, 1, 0).T, M).T

                if self.kernel == 'tricube':
                    # chosen bandwidth to hold the constrain dist/h_m <= 1
                    h_Q = dist[ind_knn[self.k_Q - 1]]
                    wei_Q = (1 - (dist[ind_knn[:self.k_Q]] / h_Q) ** 3) ** 3
                else:
                    wei_Q = wei[:self.k_Q]
                wei_Q = wei_Q / np.sum(wei_Q)

                cov_xf = np.cov(res[:, :self.k_Q],
                                aweights=wei_Q)  # ((res[:,:LLR.k_Q].dot(np.diag(wei_Q))).dot(res[:,:LLR.k_Q].T))/(1-sum(wei_Q**2));
                if (self.Q.form == 'full'):
                    Q_xf[:, :, i] = cov_xf
                elif (self.Q.form == 'diag'):
                    Q_xf[:, :, i] = np.diag(np.diag(cov_xf))
                else:
                    Q_xf[:, :, i] = np.trace(cov_xf) * self.Q.base / dx

            else:
                Q_xf[:, :, i] = self.Q.value

        # %% LLR sampling
        for i in range(N):
            if len(ind_x) > 1:
                xf[:, i] = np.random.multivariate_normal(mean_xf[:, ind_x[i]],
                                                         Q_xf[:, :, ind_x[i]])
            else:
                xf[:, i] = np.random.multivariate_normal(
                    mean_xf[:, i], Q_xf[:, :, i])

        return xf, mean_xf, Q_xf, M_xf
