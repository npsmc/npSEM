#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 13:21:13 2018

@author: trang
"""
import matplotlib.pyplot as plt  # plot
import numpy as np
from matplotlib.ticker import LinearLocator, FormatStrFormatter


def k_choice(LLR, x, y):
    dx, T = x.shape;
    if LLR.Q.type == 'fixed':
        L = np.zeros((len(LLR.nN_m), 1));
        E = np.zeros((len(LLR.nN_m), 1))

        for i in range(len(LLR.nN_m)):
            LLR.k_m = LLR.nN_m[i];
            loglik = 0;
            err = 0
            for t in range(T):
                xf, mean_xf, Q_xf, M_xf = m_LLR(x[:, t][np.newaxis].T, t, np.ones([1]), LLR);
                innov = y[:, t][np.newaxis].T - mean_xf
                loglik += .5 * np.log(2 * np.pi * np.linalg.det(np.squeeze(Q_xf)))
                loglik -= .5 * innov.T.dot(np.linalg.inv(np.squeeze(Q_xf))).dot(innov)
                err += np.sqrt(np.mean((y[:, t][np.newaxis].T - mean_xf) ** 2))
            L[i] = loglik;
            E[i] = err;
        ind_max = np.argmax(L)
        k_m = LLR.nN_m[ind_max];
        k_Q = []
    #
    #        plt.figure(3)
    #        plt.rcParams['figure.figsize'] = (8, 5)
    #        fig, ax1 = plt.subplots()
    #        ax2 = ax1.twinx()
    #        ax1.plot(LLR.nN_m, L, color='r')
    #        ax1.set_xlabel('number of $m$-analogs $(k_m)$')
    #        ax1.set_ylabel('log likelihood')
    #
    #        ax2.plot(LLR.nN_m, E, color='b')
    #        ax2.set_ylabel('RMSE')
    #        plt.grid()
    #        plt.show()
    else:
        L = np.zeros((len(LLR.nN_m), len(LLR.nN_Q)));
        E = np.zeros((len(LLR.nN_m), len(LLR.nN_Q)))

        for i in range(len(LLR.nN_m)):
            LLR.k_m = LLR.nN_m[i];
            for j in range(len(LLR.nN_Q)):
                loglik = 0;
                err = 0
                LLR.k_Q = LLR.nN_Q[j];
                for t in range(T):
                    xf, mean_xf, Q_xf, M_xf = m_LLR(x[:, t][np.newaxis].T, t, np.ones([1]), LLR);
                    innov = y[:, t][np.newaxis].T - mean_xf
                    loglik += .5 * np.log(2 * np.pi * np.linalg.det(np.squeeze(Q_xf)))
                    loglik -= .5 * innov.T.dot(np.linalg.inv(np.squeeze(Q_xf))).dot(innov)
                    err += np.sqrt(np.mean((y[:, t][np.newaxis].T - mean_xf) ** 2))
                L[i, j] = loglik;
                E[i, j] = err
        ind_max = np.unravel_index(np.argmax(L, axis=None), L.shape)
        k_m = LLR.nN_m[ind_max[0]];
        k_Q = LLR.nN_Q[ind_max[1]];
        print(L);
        print(E)
        fig = plt.figure(3)
        plt.rcParams['figure.figsize'] = (5, 8)

        ax = fig.gca(projection='3d')
        Y, X = np.meshgrid(LLR.nN_Q, LLR.nN_m)
        surf = ax.plot_surface(X, Y, L, cmap='Reds',
                               linewidth=0, antialiased=False)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        fig.colorbar(surf, shrink=0.5, aspect=5)
        ax.set_xlabel('number of $m$-analogs ($k_m$)')
        ax.set_ylabel('number of $Q$-analogs ($k_Q$)')
        ax.set_zlabel('log likelihood')
        plt.grid()
        fig = plt.figure(4)
        plt.rcParams['figure.figsize'] = (5, 8)
        ax = fig.gca(projection='3d')
        Y, X = np.meshgrid(LLR.nN_Q, LLR.nN_m)
        surf = ax.plot_surface(X, Y, E, cmap='Blues',
                               linewidth=0, antialiased=False)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        fig.colorbar(surf, shrink=0.5, aspect=5)
        ax.set_xlabel('number of analogs $(k_m)$')
        ax.set_ylabel('number of analogs $(k_Q)$')
        ax.set_zlabel('RMSE')
        plt.grid()
    #        plt.show()
    # cm.coolwarm
    # Add a color bar which maps values to colors.

    return k_m, k_Q
