"""
* Author:  Michael Barber
* File:    linear_lib.py
* Purpose: This module provides utilities for performing linear regression
*          predictions on wine quality data data file(s) of wine 
*          characteristics obtained from 
*          <http://archive.ics.uci.edu/ml/datasets/Wine+Quality>
"""

import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import sys
import random

# Function for defining a standardize and an unStandardize function.
# note: Leveraged makeStandardize from:
# <http://www.cs.colostate.edu/~anderson/cs545/index.html/doku.php?id
# =notes:notes3a>
def makeStandardize(X):
    means = X.mean(axis=0)
    stds = X.std(axis=0)
    def standardize(origX):
        return (origX - means) / stds
    def unStandardize(stdX):
        return stds * stdX + means
    return (standardize, unStandardize)

# Function that calculates linear least-square & returns weights
def makeLLS(independent, dependent):
    w = np.linalg.solve(np.dot(independent.T, independent), \
                        np.dot(independent.T, dependent))
    return w

# Function that returns a prediction of the target variables
def useLLS(weights, independent):
    predict = np.dot(independent, weights)
    return predict

def printWeights(weights, names):
    sortedOrder = abs(weights).argsort(axis=0).flatten()
    #reverse for indices from largest to smallest
    sortedOrder = sortedOrder[::-1]
    for (ni,wi) in zip(np.array(names)[sortedOrder], weights[sortedOrder]):
        print '%15s %4.4f' % (ni,wi[0])

def plotPredictions(prediction, dependent):
    maxQ = max(min(prediction), min(dependent))
    minQ = min(max(prediction), max(dependent))

    plt.clf()
    plt.plot(prediction, dependent, 'o')
    plt.plot([maxQ,minQ],[maxQ,minQ], 'r',linewidth=3)
    plt.xlabel("Predicted Quality")
    plt.ylabel("Actual Quality")
    plt.show()
#    plt.savefig(output_filename)

def calculateRMSE(weights, independent, dependent):
    rmse = sqrt(np.mean((np.dot(independent, weights) - dependent)**2))
    return rmse

