
# coding: utf-8

# # Lab 3: Bayes Classifier and Boosting

# ## Jupyter notebooks
#
# In this lab, you can use Jupyter <https://jupyter.org/> to get a nice layout of your code and plots in one document. However, you may also use Python as usual, without Jupyter.
#
# If you have Python and pip, you can install Jupyter with `sudo pip install jupyter`. Otherwise you can follow the instruction on <http://jupyter.readthedocs.org/en/latest/install.html>.
#
# And that is everything you need! Now use a terminal to go into the folder with the provided lab files. Then run `jupyter notebook` to start a session in that folder. Click `lab3.ipynb` in the browser window that appeared to start this very notebook. You should click on the cells in order and either press `ctrl+enter` or `run cell` in the toolbar above to evaluate all the expressions.

# ## Import the libraries
#
# Check out `labfuns.py` if you are interested in the details.

import sys
import numpy
from math import log as ln
import numpy as np
from scipy import misc
from imp import reload
from labfuns import *
from sklearn import decomposition
from matplotlib.colors import ColorConverter

def fst((a,b)): return a
def snd((a,b)): return b
def fix(f): return lambda a: lambda b: f(b)(a)
def take(i): return lambda l: l[:i]
def eq(a): return lambda b: a == b
def div(a): return lambda b: a / b
def compose(f,g): return lambda x: f(g(x))
def mean(x): return sum(x)/len(x)

# ## Bayes classifier functions to implement
#
# The lab descriptions state what each function should do.

# Note that you do not need to handle the W argument for this part
# in: labels - N x 1 vector of class labels
# out: prior - C x 1 vector of class priors
def computePrior(labels,W=None):
    ks=set(labels)
    N=len(labels)
    if W is None:
        W = numpy.array([1.0/N] * N)
    weightedLabels = zip(W,labels)
    getClass = snd
    getWeight = fst
    ofClass = lambda k: compose(eq(k),getClass)
    weightsByClass = lambda k: map(getWeight, filter(ofClass(k), weightedLabels))
    prior = map(compose(sum,weightsByClass), ks)
    return numpy.array(prior)

def sig((muk,WXk)):
    getWeight = fst
    getX = snd
    Wk = map(getWeight, WXk)
    Xk = map(getX, WXk)
    diff = Xk-muk
    return numpy.dot(Wk * diff.T, diff) / sum(Wk)

def weightedmean(xs):
    getWeight = fst
    weightValue = lambda (wi,xi): wi*xi
    m = sum(map(weightValue,xs))/sum(map(getWeight,xs))
    return m

# Note that you do not need to handle the W argument for this part
# in:      X - N x d matrix of N data points
#     labels - N x 1 vector of class labels
# out:    mu - C x d matrix of class means
#      sigma - d x d x C matrix of class covariances
def mlParams(X,labels,W=None):
    ks = set(labels)
    N = len(X)
    if W is None:
        W = numpy.array([1.0/N] * N)
    getClass = fst
    getValue = snd
    weightedValues = zip(W,X)
    labledValues = zip(labels,weightedValues)
    ofClass = lambda k: compose(eq(k),getClass)
    getXOfClass = lambda k: map(getValue, filter(ofClass(k), labledValues))
    wxsByClass = map(getXOfClass, ks) #[pairs of (wi,xi)] grouped by class
    mu = numpy.array(map(weightedmean, wxsByClass))
    sigma = numpy.array(map(sig,zip(mu,wxsByClass))).T
    return mu, sigma

# in:      X - N x d matrix of M data points
#      prior - C x 1 vector of class priors
#         mu - C x d matrix of class means
#      sigma - d x d x C matrix of class covariances
# out:     h - N x 1 class predictions for test points
def classify(X,prior,mu,sigma,covdiag=True):
    h = [0] * len(X)
    Ls = [np.linalg.cholesky(sigma[:,:,k]) for k in range(len(mu))]
    for i,x_star in enumerate(X):
        maxval = float('-inf')
        selectedClass = -1
        for k, muk in enumerate(mu):
            sigma_k = sigma[:,:,k]
            if covdiag==True:
                diff = x_star-muk
                y = np.linalg.solve(sigma_k,diff.T)
            else:
                L = Ls[k]
                diff = x_star-muk
                v = np.linalg.solve(L,diff.T)
                y = np.linalg.solve(L.T,v)
            (sign, logdet) = numpy.linalg.slogdet(sigma_k)
            a = -0.5 * sign * logdet
            b = -0.5 * numpy.dot(diff,y)
            c = ln(prior[k])
            total = a + b + c
            if total > maxval:
                selectedClass = k
                maxval = total
        h[i] = selectedClass
    return numpy.array(h)

# ## Test the Maximum Likelihood estimates
#
# Call `genBlobs` and `plotGaussian` to verify your estimates.

def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d()
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)

X, labels = genBlobs(centers=5)
N = len(X)
W = numpy.array(map(lambda (i,x): i+1, enumerate(X)))
W = W / np.linalg.norm(W)
mu, sigma = mlParams(X,labels,W)
prior = computePrior(labels,W)
# plotGaussian(X,labels,mu,sigma)


# ## Boosting functions to implement
#
# The lab descriptions state what each function should do.

# in:       X - N x d matrix of N data points
#      labels - N x 1 vector of class labels
#           T - number of boosting iterations
# out: priors - length T list of prior as above
#         mus - length T list of mu as above
#      sigmas - length T list of sigma as above
#      alphas - T x 1 vector of vote weights
def trainBoost(X,labels,T=5,covdiag=True):
    N = len(X)
    W = numpy.array([1/N] * N)
    iteration = 0
    priors = computePrior(labels, W)
    mus, sigmas = mlParams(X, labels, W)
    return priors,mus,sigmas,alphas

# in:       X - N x d matrix of N data points
#      priors - length T list of prior as above
#         mus - length T list of mu as above
#      sigmas - length T list of sigma as above
#      alphas - T x 1 vector of vote weights
# out:  yPred - N x 1 class predictions for test points
def classifyBoost(X,priors,mus,sigmas,alphas,covdiag=True):
    # Your code here
    return c


# ## Define our testing function
#
# The function below, `testClassifier`, will be used to try out the different datasets. `fetchDataset` can be provided with any of the dataset arguments `wine`, `iris`, `olivetti` and `vowel`. Observe that we split the data into a **training** and a **testing** set.

np.set_printoptions(threshold=np.nan)
np.set_printoptions(precision=25)
np.set_printoptions(linewidth=200)

def testClassifier(dataset='iris',dim=0,split=0.7,doboost=False,boostiter=5,covdiag=True,ntrials=100):

    X,y,pcadim = fetchDataset(dataset)

    means = np.zeros(ntrials,);

    for trial in range(ntrials):

        # xTr,yTr,xTe,yTe,trIdx,teIdx = trteSplit(X,y,split)
        xTr,yTr,xTe,yTe,trIdx,teIdx = trteSplitEven(X,y,split)

        # Do PCA replace default value if user provides it
        if dim > 0:
            pcadim = dim
        if pcadim > 0:
            pca = decomposition.PCA(n_components=pcadim)
            pca.fit(xTr)
            xTr = pca.transform(xTr)
            xTe = pca.transform(xTe)

        ## Boosting
        if doboost:
            # Compute params
            priors,mus,sigmas,alphas = trainBoost(xTr,yTr,T=boostiter)
            yPr = classifyBoost(xTe,priors,mus,sigmas,alphas,covdiag=covdiag)
        else:
        ## Simple
            # Compute params
            prior = computePrior(yTr)
            mu, sigma = mlParams(xTr,yTr)
            # Predict
            yPr = classify(xTe,prior,mu,sigma,covdiag=covdiag)

        # Compute classification error
        # print "Trial:",trial,"Accuracy",100*np.mean((yPr==yTe).astype(float))

        means[trial] = 100*np.mean((yPr==yTe).astype(float))

    print "Final mean classification accuracy ", np.mean(means), "with standard deviation", np.std(means)


# ## Plotting the decision boundary
#
# This is some code that you can use for plotting the decision boundary
# boundary in the last part of the lab.

def plotBoundary(dataset='iris',split=0.7,doboost=False,boostiter=5,covdiag=True):

    X,y,pcadim = fetchDataset(dataset)
    xTr,yTr,xTe,yTe,trIdx,teIdx = trteSplitEven(X,y,split)
    pca = decomposition.PCA(n_components=2)
    pca.fit(xTr)
    xTr = pca.transform(xTr)
    xTe = pca.transform(xTe)

    pX = np.vstack((xTr, xTe))
    py = np.hstack((yTr, yTe))

    if doboost:
        ## Boosting
        # Compute params
        priors,mus,sigmas,alphas = trainBoost(xTr,yTr,T=boostiter,covdiag=covdiag)
    else:
        ## Simple
        # Compute params
        prior = computePrior(yTr)
        mu, sigma = mlParams(xTr,yTr)

    xRange = np.arange(np.min(pX[:,0]),np.max(pX[:,0]),np.abs(np.max(pX[:,0])-np.min(pX[:,0]))/100.0)
    yRange = np.arange(np.min(pX[:,1]),np.max(pX[:,1]),np.abs(np.max(pX[:,1])-np.min(pX[:,1]))/100.0)

    grid = np.zeros((yRange.size, xRange.size))

    for (xi, xx) in enumerate(xRange):
        for (yi, yy) in enumerate(yRange):
            if doboost:
                ## Boosting
                grid[yi,xi] = classifyBoost(np.matrix([[xx, yy]]),priors,mus,sigmas,alphas,covdiag=covdiag)
            else:
                ## Simple
                grid[yi,xi] = classify(np.matrix([[xx, yy]]),prior,mu,sigma,covdiag=covdiag)

    classes = range(np.min(y), np.max(y)+1)
    ys = [i+xx+(i*xx)**2 for i in range(len(classes))]
    colormap = cm.rainbow(np.linspace(0, 1, len(ys)))

    plt.hold(True)
    conv = ColorConverter()
    for (color, c) in zip(colormap, classes):
        try:
            CS = plt.contour(xRange,yRange,(grid==c).astype(float),15,linewidths=0.25,colors=conv.to_rgba_array(color))
        except ValueError:
            pass
        xc = pX[py == c, :]
        plt.scatter(xc[:,0],xc[:,1],marker='o',c=color,s=40,alpha=0.5)

    plt.xlim(np.min(pX[:,0]),np.max(pX[:,0]))
    plt.ylim(np.min(pX[:,1]),np.max(pX[:,1]))


# ## Run some experiments
#
# Call the `testClassifier` and `plotBoundary` functions for this part.

# Example usage of the functions

for s in ['iris']: #['iris','wine','olivetti','vowel']:
    print('With covdiag:')
    testClassifier(dataset=s,split=0.7,doboost=True,boostiter=5,covdiag=True)
    print('Without covdiag:')
    testClassifier(dataset=s,split=0.7,doboost=True,boostiter=5,covdiag=False)
    print('')
    # plotBoundary(dataset=s,split=0.7,doboost=False,boostiter=5,covdiag=True)
    # plotBoundary(dataset=s,split=0.7,doboost=False,boostiter=5,covdiag=False)
    # plt.show()
