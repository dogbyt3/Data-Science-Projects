"""
* Author:  Michael Barber
* File:    linear_lib.py

"""

import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import csv
import sys
import random
from math import ceil, floor

######################################################################
### Linear model functions
def makeStandardize(X):
    means = X.mean(axis=0)
    stds = X.std(axis=0)
    def standardize(origX):
        return (origX - means) / stds
    def unStandardize(stdX):
        return stds * stdX + means
    return (standardize, unStandardize)

def makeLLS(X,T):
    (standardizeF, unstandardizeF) = makeStandardize(X)
    X = standardizeF(X)
    (nRows,nCols) = X.shape
    X = np.hstack((np.ones((X.shape[0],1)), X))    
    w = np.linalg.solve(np.dot(X.T,X), np.dot(X.T,T))
    return (w, standardizeF, unstandardizeF)

def useLLS(w,standardizeF,X):
    X = standardizeF(X)
    X = np.hstack((np.ones((X.shape[0],1)), X))
    return np.dot(X,w)

def calculateRMSE(weights, independent, dependent):
    rmse = sqrt(np.mean((np.dot(independent, weights) - dependent)**2))
    return rmse

def printWeights(Wall, lowerW, upperW, nAttributes):
    print "Weights:"
    for wi in range(nAttributes):
        print "weight %d median= %.3f range= %.3f to %.3f" % (wi, np.median(Wall[wi,:]),lowerW[wi],upperW[wi])

######################################################################

def calculateConfidenceLevels(X, T, trainingFraction, nModels, confidenceLevel,
                              nData, nTest):

    ### Plot the initial data generated
    plt.figure(1)
    plt.clf()
    plt.subplot(4,1,1)
    plt.plot(X,T,'o-')

    ### Create Test Data Set
    # test data evenly spaces along x axis between -5 and 15
    Xtest = np.linspace(-5,15,nTest).reshape((nTest,1))
    Xtest = np.hstack((np.ones((Xtest.shape[0],1)), Xtest))

    nRows = X.shape[0]
    X = np.hstack((np.ones((nRows,1)), X))

    Yall = np.zeros((nTest,nModels))
    Wall = np.zeros((X.shape[1],nModels))

    # Make bootstrapped models
    for modeli in range(nModels):
        trainI = [random.randint(0,nRows-1) for i in xrange(nRows)]
        Xtrain = X[trainI,:]
        Ttrain = T[trainI,:]
        w = np.linalg.solve(np.dot(Xtrain.T,Xtrain), np.dot(Xtrain.T,Ttrain))
        Wall[:,modeli] = w.flatten()
        Ytest = np.dot(Xtest, w)
        Yall[:,modeli] = Ytest.flatten()

    # plot the predictions and models 
    plt.subplot(4,1,2)
    plt.plot(X[:,1:],T,'o-')
    plt.plot(Xtest[:,1:],Yall)

    # create the confidence intervals from the models
    confidenceFraction = (1-confidenceLevel/100.0)/2.0
    loweri = int(ceil(nModels*confidenceFraction))
    upperi = int(floor(nModels*(1-confidenceFraction)))

    lower = np.zeros((nTest))
    upper = np.zeros((nTest))
    for xi in range(nTest):
        predictions = Yall[xi,:]
        predictions.sort()
        lower[xi] = predictions[loweri]
        upper[xi] = predictions[upperi]

        nAttributes = X.shape[1]
        lowerW = np.zeros((nAttributes))
        upperW = np.zeros((nAttributes))
        for wi in range(nAttributes):
            wis = Wall[wi,:]
            wis.sort()
            lowerW[wi] = wis[loweri]
            upperW[wi] = wis[upperi]

    print "Weights:"
    for wi in range(nAttributes):
        print "weight %d median= %.3f range= %.3f to %.3f" % \
            (wi, np.median(Wall[wi,:]),lowerW[wi],upperW[wi])

    # plot the confidence intervals
    plt.subplot(4,1,3)
    plt.plot(X[:,1:],T,'o-')
    plt.fill_between(Xtest[:,1:].flatten(),lower,upper,alpha=0.4)
            
    #plot the distribution of weight values versus attribute
    plt.subplot(4,1,4)
    plt.boxplot(Wall.T)
    plt.xlabel("Attribute")
    plt.ylabel("Distribution of weight values")
    plt.show()
