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
import csv
import sys
import random

###############################################################################
#   Input variables (based on physicochemical tests):
#   1 - fixed acidity
#   2 - volatile acidity
#   3 - citric acid
#   4 - residual sugar
#   5 - chlorides
#   6 - free sulfur dioxide
#   7 - total sulfur dioxide
#   8 - density
#   9 - pH
#   10 - sulphates
#   11 - alcohol
#   Output variable (based on sensory data): 
#   12 - quality (score between 0 and 10)
#
#   Missing Attribute Values: None
###############################################################################
names = ['f_acidity', 'v_acidity', 'citric_acid', 'sugar', 'chlorides',\
         'free_sulfur', 'total_sulfur', 'density', 'pH', 'sulphates', \
         'alcohol', 'quality']

def open_file(filename):
    #open the file for parsing
    try:
        data = np.loadtxt(filename, usecols=range(12),skiprows=1, delimiter=';')
        return data
    except IOError:
        print 'Cannot open filename: \'%s\' for reading. <File Not Found!>' \
            % filename
        exit(2)

# function used to plot initial, nonstandardized data
def plot_data(dependent_variable, independent_variables, \
              dependent_variable_name, independent_variable_names, \
              output_filename):

    plt.figure(2)
    plt.clf()

    # plot the independent variables against the quality
    for c in range(independent_variables.shape[1]):
        plt.subplot(3,4,c+1)
        plt.plot(dependent_variable, independent_variables[:,c], '.')
        plt.ylabel(independent_variable_names[c])
        plt.xlabel(dependent_variable_name)

    plt.show()
    #plt.savefig(output_filename)

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

def prepareDataForModel(original_data, set_percent):
    # create column vector of quality (dependent variable)  
    dependent_data = original_data[:,11:12]

    # create matrix of remaining independent variables
    independent_data = original_data[:,:11]    
    independent_names = names[:11]
    dependent_name = names[11]

    # indices for all data rows
    (nrow,ncol) = independent_data.shape
    all_indices = xrange(nrow)

    # number of training samples
    training_set_size = int(round(nrow*set_percent))
    testing_set_size = nrow-training_set_size

    # row indices for training samples
    training_set_indices = \
        list(set(random.sample(all_indices,training_set_size))) 

    # row indices for testing samples
    testing_set_indices = \
        list(set(all_indices).difference(set(training_set_indices)))

    # create training sets
    training_set_dependent = dependent_data[training_set_indices,:]
    training_set_independent = independent_data[training_set_indices,:]

    # create test sets
    testing_set_dependent = dependent_data[testing_set_indices,:]
    testing_set_independent = independent_data[testing_set_indices,:]

    # standardize the data sets
    (standardize, unStandardize) = makeStandardize(training_set_independent)
    training_set_independent_std = standardize(training_set_independent)
    testing_set_independent_std =  standardize(testing_set_independent)

    # add a '1' to first attribute of every sample
    training_set_independent_std = \
        np.hstack((np.ones((training_set_size,1)),training_set_independent_std))

    testing_set_independent_std = \
        np.hstack((np.ones((testing_set_size,1)),testing_set_independent_std))

    # add the bias to the list of independent variables
    independent_names.insert(0, 'bias')

    return (training_set_independent_std, training_set_dependent,
            testing_set_independent_std, testing_set_dependent,
            dependent_name, independent_names)

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

