# Source: https://github.com/jtrive84

"""
Local Regression (LOESS) estimation routine.
"""
import numpy as np
import pandas as pd
import scipy


def loc_eval(x, b):
    """
    Evaluate `x` using locally-weighted regression parameters.
    Degree of polynomial used in loess is inferred from b. `x`
    is assumed to be a scalar.
    """
    loc_est = 0
    for i in enumerate(b): loc_est+=i[1]*(x**i[0])
    return(loc_est)



def loess(xvals, yvals, alpha, poly_degree=1):
    """
    Perform locally-weighted regression on xvals & yvals.
    Variables used inside `loess` function:

        n         => number of data points in xvals
        m         => nbr of LOESS evaluation points
        q         => number of data points used for each
                     locally-weighted regression
        v         => x-value locations for evaluating LOESS
        locsDF    => contains local regression details for each
                     location v
        evalDF    => contains actual LOESS output for each v
        X         => n-by-(poly_degree+1) design matrix
        W         => n-by-n diagonal weight matrix for each
                     local regression
        y         => yvals
        b         => local regression coefficient estimates.
                     b = `(X^T*W*X)^-1*X^T*W*y`. Note that `@`
                     replaces `np.dot` in recent numpy versions.
        local_est => response for local regression
    """
    # Sort dataset by xvals.
    all_data = sorted(zip(xvals, yvals), key=lambda x: x[0])
    xvals, yvals = zip(*all_data)

    locsDF = pd.DataFrame(
                columns=[
                  'loc','x','weights','v','y','raw_dists',
                  'scale_factor','scaled_dists'
                  ])
    evalDF = pd.DataFrame(
                columns=[
                  'loc','est','b','v','g'
                  ])

    n = len(xvals)
    m = n + 1
    q = int(np.floor(n * alpha) if alpha <= 1.0 else n)
    avg_interval = ((max(xvals)-min(xvals))/len(xvals))
    v_lb = max(0,min(xvals)-(.5*avg_interval))
    v_ub = (max(xvals)+(.5*avg_interval))
    v = enumerate(np.linspace(start=v_lb, stop=v_ub, num=m), start=1)

    # Generate design matrix based on poly_degree.
    xcols = [np.ones_like(xvals)]
    for j in range(1, (poly_degree + 1)):
        xcols.append([i ** j for i in xvals])
    X = np.vstack(xcols).T


    for i in v:

        iterpos = i[0]
        iterval = i[1]

        # Determine q-nearest xvals to iterval.
        iterdists = sorted([(j, np.abs(j-iterval)) \
                           for j in xvals], key=lambda x: x[1])

        _, raw_dists = zip(*iterdists)

        # Scale local observations by qth-nearest raw_dist.
        scale_fact = raw_dists[q-1]
        scaled_dists = [(j[0],(j[1]/scale_fact)) for j in iterdists]
        weights = [(j[0],((1-np.abs(j[1]**3))**3 \
                      if j[1]<=1 else 0)) for j in scaled_dists]

        # Remove xvals from each tuple:
        _, weights      = zip(*sorted(weights,     key=lambda x: x[0]))
        _, raw_dists    = zip(*sorted(iterdists,   key=lambda x: x[0]))
        _, scaled_dists = zip(*sorted(scaled_dists,key=lambda x: x[0]))

        iterDF1 = pd.DataFrame({
                    'loc'         :iterpos,
                    'x'           :xvals,
                    'v'           :iterval,
                    'weights'     :weights,
                    'y'           :yvals,
                    'raw_dists'   :raw_dists,
                    'scale_fact'  :scale_fact,
                    'scaled_dists':scaled_dists
                    })

        locsDF    = pd.concat([locsDF, iterDF1])
        W         = np.diag(weights)
        y         = yvals
        b         = np.linalg.inv(X.T @ W @ X) @ (X.T @ W @ y)
        local_est = loc_eval(iterval, b)
        iterDF2   = pd.DataFrame({
                       'loc':[iterpos],
                       'b'  :[b],
                       'v'  :[iterval],
                       'g'  :[local_est]
                       })

        evalDF = pd.concat([evalDF, iterDF2])

    # Reset indicies for returned DataFrames.
    locsDF.reset_index(inplace=True)
    locsDF.drop('index', axis=1, inplace=True)
    locsDF['est'] = 0; evalDF['est'] = 0
    locsDF = locsDF[['loc','est','v','x','y','raw_dists',
                     'scale_fact','scaled_dists','weights']]

    # Reset index for evalDF.
    evalDF.reset_index(inplace=True)
    evalDF.drop('index', axis=1, inplace=True)
    evalDF = evalDF[['loc','est', 'v', 'b', 'g']]

    return(locsDF, evalDF)

# +
"""
Calling `loess` on a dataset that follows a known functional 
form with added random perturbations.
"""
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(16)

xvals = [.39,.75,1.1,1.45,1.95,2.46,3.07,3.44,4.57,5.05,5.68,
         6.01,6.63,7.11,7.62,8.01,8.54,9.08,9.48, 9.91]

yvals = [1.25*np.sqrt(i)+np.random.normal(0, .425) for i in xvals]

# loess returns a tuple of DataFrames, named here as `regsDF` and
# `evalDF` for "Regression DataFrame" and "Evaluation DataFrame":
regsDF, evalDF = loess(xvals, yvals, alpha=.6, poly_degree=1)

# +
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(context='notebook', style='darkgrid', font_scale=1)

np.random.seed(16)

xvals = [.39,.75,1.1,1.45,1.95,2.46,3.07,3.44,4.57,5.05,5.68,
         6.01,6.63,7.11,7.62,8.01,8.54,9.08,9.48, 9.91]

yvals = [1.25*np.sqrt(i)+np.random.normal(0, .425) for i in xvals]

regsDF, evalDF = loess(xvals, yvals, alpha=.6, poly_degree=1)

# Obtain reference to LOESS x & y values (v & g).
l_x  = evalDF['v'].values
l_y  = evalDF['g'].values

# Generate x-y scatterplot with loess estimate overlaid.
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.grid(True)
ax1.minorticks_on()
ax1.scatter(xvals, yvals, label="Original Data")
ax1.plot(l_x, l_y, color='#FF0000', label="1st-degree Polynomial LOESS")
ax1.set_title("Linear LOESS Estimator", loc="left", fontsize=14)
ax1.legend(loc="upper left",
           scatterpoints=1,
           fontsize=11,
           frameon=True,
           fancybox=True,
           facecolor="#FFFFFF",
           edgecolor="#000000")
plt.tight_layout()

# +
regsDF1, evalDF1 = loess(xvals, yvals, alpha=.6, poly_degree=1)
regsDF2, evalDF2 = loess(xvals, yvals, alpha=.6, poly_degree=2)

l_x1  = evalDF['v'].values
l_y1  = evalDF['g'].values

l_x2 = evalDF2['v'].values
l_y2 = evalDF2['g'].values


# `poly_degree=1` plot:
fig = plt.figure()
ax1 = fig.add_subplot(121)
ax1.grid(True)
ax1.minorticks_on()
ax1.scatter(xvals, yvals, label="Original Data")
ax1.plot(l_x1, l_y1, color='#FF0000', label="1st-degree Polynomial LOESS")
ax1.set_title("Linear LOESS Estimator", loc="left", fontsize=14)
ax1.legend(loc="upper left",
           scatterpoints=1,
           fontsize=11,
           frameon=True,
           fancybox=True,
           facecolor="#FFFFFF",
           edgecolor="#000000")

# `poly_degree=2` plot:
ax2 = fig.add_subplot(122)
ax2.grid(True)
ax2.minorticks_on()
ax2.scatter(xvals, yvals, label="Original Data")
ax2.plot(l_x2, l_y2, color='#FF0000', label="2nd-degree Polynomial LOESS")
ax2.set_title("Quadratic LOESS Estimator", loc="left", fontsize=14)
ax2.legend(loc="upper left",
           scatterpoints=1,
           fontsize=11,
           frameon=True,
           fancybox=True,
           facecolor="#FFFFFF",
           edgecolor="#000000")

plt.tight_layout()

# +
"""
Local Regression (LOESS) estimation routine with optional 
iterative robust estimation procedure. Setting `robustify=True` 
indicates that the robust estimation procedure should be 
performed. 
"""
import numpy as np
import pandas as pd
import scipy


def loc_eval(x, b):
    """
    Evaluate `x` using locally-weighted regression parameters.
    Degree of polynomial used in loess is inferred from b. `x`
    is assumed to be a scalar.
    """
    loc_est = 0
    for i in enumerate(b): loc_est+=i[1]*(x**i[0])
    return(loc_est)



def loess(xvals, yvals, alpha, poly_degree=1, robustify=False):
    """
    Perform locally-weighted regression via xvals & yvals.
    Variables used within `loess` function:

        n         => number of data points in xvals
        m         => nbr of LOESS evaluation points
        q         => number of data points used for each
                     locally-weighted regression
        v         => x-value locations for evaluating LOESS
        locsDF    => contains local regression details for each
                     location v
        evalDF    => contains actual LOESS output for each v
        X         => n-by-(poly_degree+1) design matrix
        W         => n-by-n diagonal weight matrix for each
                     local regression
        y         => yvals
        b         => local regression coefficient estimates.
                     b = `(X^T*W*X)^-1*X^T*W*y`. Note that `@`
                     replaces np.dot in recent numpy versions.
        local_est => response for local regression
    """
    # sort dataset by xvals:
    all_data = sorted(zip(xvals, yvals), key=lambda x: x[0])
    xvals, yvals = zip(*all_data)

    locsDF = pd.DataFrame(
                columns=[
                  'loc','x','weights','v','y','raw_dists',
                  'scale_factor','scaled_dists'
                  ])
    evalDF = pd.DataFrame(
                columns=[
                  'loc','est','b','v','g'
                  ])

    n = len(xvals)
    m = n + 1
    q = int(np.floor(n * alpha) if alpha <= 1.0 else n)
    avg_interval = ((max(xvals)-min(xvals))/len(xvals))
    v_lb = max(0,min(xvals)-(.5*avg_interval))
    v_ub = (max(xvals)+(.5*avg_interval))
    v = enumerate(np.linspace(start=v_lb, stop=v_ub, num=m), start=1)

    # Generate design matrix based on poly_degree.
    xcols = [np.ones_like(xvals)]
    for j in range(1, (poly_degree + 1)):
        xcols.append([i ** j for i in xvals])
    X = np.vstack(xcols).T


    for i in v:

        iterpos = i[0]
        iterval = i[1]

        # Determine q-nearest xvals to iterval.
        iterdists = sorted([(j, np.abs(j-iterval)) \
                           for j in xvals], key=lambda x: x[1])

        _, raw_dists = zip(*iterdists)

        # Scale local observations by qth-nearest raw_dist.
        scale_fact = raw_dists[q-1]
        scaled_dists = [(j[0],(j[1]/scale_fact)) for j in iterdists]
        weights = [(j[0],((1-np.abs(j[1]**3))**3 \
                      if j[1]<=1 else 0)) for j in scaled_dists]

        # Remove xvals from each tuple:
        _, weights      = zip(*sorted(weights,     key=lambda x: x[0]))
        _, raw_dists    = zip(*sorted(iterdists,   key=lambda x: x[0]))
        _, scaled_dists = zip(*sorted(scaled_dists,key=lambda x: x[0]))

        iterDF1 = pd.DataFrame({
                    'loc'         :iterpos,
                    'x'           :xvals,
                    'v'           :iterval,
                    'weights'     :weights,
                    'y'           :yvals,
                    'raw_dists'   :raw_dists,
                    'scale_fact'  :scale_fact,
                    'scaled_dists':scaled_dists
                    })

        locsDF = pd.concat([locsDF, iterDF1])
        W = np.diag(weights)
        y = yvals
        b = np.linalg.inv(X.T @ W @ X) @ (X.T @ W @ y)
        local_est = loc_eval(iterval, b)

        iterDF2 = pd.DataFrame({
                     'loc':[iterpos],
                     'b'  :[b],
                     'v'  :[iterval],
                     'g'  :[local_est]
                     })

        evalDF = pd.concat([evalDF, iterDF2])

    # Reset indicies for returned DataFrames.
    locsDF.reset_index(inplace=True)
    locsDF.drop('index', axis=1, inplace=True)
    locsDF['est'] = 0; evalDF['est'] = 0
    locsDF = locsDF[['loc','est','v','x','y','raw_dists',
                     'scale_fact','scaled_dists','weights']]


    if robustify==True:

        cycle_nbr = 1
        robust_est = [evalDF]

        while True:
            # Perform iterative robustness procedure for each local regression.
            # Evaluate local regression for each item in xvals.
            #
            # e1_i => raw residuals
            # e2_i => scaled residuals
            # r_i  => robustness weight
            revalDF = pd.DataFrame(
                            columns=['loc','est','v','b','g']
                            )

            for i in robust_est[-1]['loc']:

                prevDF = robust_est[-1]
                locDF = locsDF[locsDF['loc']==i]
                b_i = prevDF.loc[prevDF['loc']==i,'b'].item()
                w_i = locDF['weights']
                v_i = prevDF.loc[prevDF['loc']==i, 'v'].item()
                g_i = prevDF.loc[prevDF['loc']==i, 'g'].item()
                e1_i = [k-loc_eval(j,b_i) for (j,k) in zip(xvals,yvals)]
                e2_i = [j/(6*np.median(np.abs(e1_i))) for j in e1_i]
                r_i = [(1-np.abs(j**2))**2 if np.abs(j)<1 else 0 for j in e2_i]
                w_f = [j*k for (j,k) in zip(w_i, r_i)]    # new weights
                W_r = np.diag(w_f)
                b_r = np.linalg.inv(X.T @ W_r @ X) @ (X.T @ W_r @ y)
                riter_est = loc_eval(v_i, b_r)

                riterDF = pd.DataFrame({
                             'loc':[i],
                             'b'  :[b_r],
                             'v'  :[v_i],
                             'g'  :[riter_est],
                             'est':[cycle_nbr]
                             })

                revalDF = pd.concat([revalDF, riterDF])
            robust_est.append(revalDF)

            # Compare `g` vals from two latest revalDF's in robust_est.
            idiffs = \
                np.abs((robust_est[-2]["g"]-robust_est[-1]["g"])/robust_est[-2]["g"])

            if ((np.all(idiffs<.005)) or cycle_nbr>50): break

            cycle_nbr+=1

        # Vertically bind all DataFrames from robust_est.
        evalDF = pd.concat(robust_est)

    evalDF.reset_index(inplace=True)
    evalDF.drop('index', axis=1, inplace=True)
    evalDF = evalDF[['loc','est', 'v', 'b', 'g']]

    return(locsDF, evalDF)

# +
"""
Graphical comparison of LOESS model estimates
at each robustness iteration.
"""
locsDF, evalDF = loess(xvals, yvals, alpha=.6, poly_degree=1, robustify=True)

resDF0 = evalDF[evalDF['est']==0]
l_x0   = resDF0['v'].values
l_y0   = resDF0['g'].values

resDF1 =  evalDF[evalDF['est']==1]
l_x1   = resDF1['v'].values
l_y1   = resDF1['g'].values

resDF3 =  evalDF[evalDF['est']==3]
l_x3   = resDF3['v'].values
l_y3   = resDF3['g'].values

resDF4 =  evalDF[evalDF['est']==evalDF['est'].max()]
l_x4   = resDF4['v'].values
l_y4   = resDF4['g'].values


fig = plt.figure()

ax0 = fig.add_subplot(221)
ax0.grid(True)
ax0.minorticks_on()
ax0.scatter(xvals, yvals, label="Original Data")
ax0.plot(l_x0, l_y0, color='#FF0000', label="1-D Polynomial LOESS Estimate")
ax0.set_title("LOESS Estimate 1", loc='left')
ax0.legend(loc="upper left",
           scatterpoints=1,
           fontsize=11,
           frameon=True,
           fancybox=True,
           facecolor="#FFFFFF",
           edgecolor="#000000")

ax1 = fig.add_subplot(222)
ax1.grid(True)
ax1.minorticks_on()
ax1.scatter(xvals, yvals, label="Original Data")
ax1.plot(l_x1, l_y1, color='#FF0000', label="1-D Polynomial LOESS Estimate")
ax1.set_title("LOESS Estimate 2", loc='left')
ax1.legend(loc="upper left",
           scatterpoints=1,
           fontsize=11,
           frameon=True,
           fancybox=True,
           facecolor="#FFFFFF",
           edgecolor="#000000")

ax2 = fig.add_subplot(223)
ax2.grid(True)
ax2.minorticks_on()
ax2.scatter(xvals, yvals, label="Original Data")
ax2.plot(l_x3, l_y3, color='#FF0000', label="1-D Polynomial LOESS Estimate")
ax2.set_title("LOESS Estimate 3", loc='left')
ax2.legend(loc="upper left",
           scatterpoints=1,
           fontsize=11,
           frameon=True,
           fancybox=True,
           facecolor="#FFFFFF",
           edgecolor="#000000")

ax3 = fig.add_subplot(224)
ax3.grid(True)
ax3.minorticks_on()
ax3.scatter(xvals, yvals, label="Original Data")
ax3.plot(l_x4, l_y4, color='#FF0000', label="1-D Polynomial LOESS Estimate")
ax3.set_title("LOESS Estimate 4", loc='left')
ax3.legend(loc="upper left",
           scatterpoints=1,
           fontsize=11,
           frameon=True,
           fancybox=True,
           facecolor="#FFFFFF",
           edgecolor="#000000")

plt.tight_layout()
plt.show()
# -


