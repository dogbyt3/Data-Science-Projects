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

def open_file(filename):
    #open the file for parsing
    try:
        data = np.loadtxt(filename, usecols=(2,3,4,6,7,8,9), delimiter=',')
        return data
    except IOError:
        print 'Cannot open filename: \'%s\' for reading. <File Not Found!>' \
            % filename
        exit(2)

def create_sets(original_data, trainingFraction):
    # create column vector for targets (dependent variable)  
    dependent_data = original_data[:,6:7]
    independent_data = original_data[:,:7]

    # indices for all data rows
    (nrow,ncol) = independent_data.shape
    all_indices = xrange(nrow)

    # number of training samples
    training_set_size = int(round(nrow*trainingFraction))
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

    return (independent_data, dependent_data, training_set_dependent,
            training_set_independent, testing_set_dependent,
            testing_set_independent)

# function used to plot initial, nonstandardized data
def plot_data(dependent_variable, independent_variables, \
              dependent_variable_name, independent_variable_names, \
              output_filename):

    plt.figure(1)
    plt.clf()

    # plot the independent variables against the quality
    for c in range(independent_variables.shape[1]):
        plt.subplot(3,4,c+1)
        plt.plot(dependent_variable, independent_variables[:,c], '.')
        plt.ylabel(independent_variable_names[c])
        plt.xlabel(dependent_variable_name)

    plt.show()

def calculateConfidenceLevels(Xdata, Tdata, Ttrain, Xtrain, Ttest, Xtest, 
                              trainingFraction, nModels, confidenceLevel,
                              nData, nTest):

    names = ['model', 'myct', 'mmin', 'mmax', 'cach', 'chmin', 'chmax', 'prp']
    ### Plot the initial data generated
    #plot_data(Tdata, Xdata, "Estimated Performance", names, "null")

    ### Divide into training and testing
    nRows = Xdata.shape[0]
    nTrain = int(round(nRows*trainingFraction))
    nTest = nRows - nTrain
    allI = range(nRows)
    trainI = random.sample(allI,nTrain)
    testI = list(set(allI).difference(set(trainI)))

    Xtrain = Xdata[trainI,:]
    Ttrain = Tdata[trainI,:]
    Xtest = Xdata[testI,:]
    Ttest = Tdata[testI,:]

    ######################################################################
    ### Make models based on bootstrap models of training data

    ### Collect all weight vectors as columns in Wall
    Wall = np.zeros((Xtrain.shape[1]+1,nModels))   
    ### Collect all standardizeF functions as elements of StandFall list
    StandFall = []
    for modeli in range(nModels):
        trainI = [random.randint(0,nTrain-1) for i in xrange(nTrain)]
        XtrainBoot = Xtrain[trainI,:]
        TtrainBoot = Ttrain[trainI,:]
        # get the weight vector and standardize function as a result
        # of making a linear least squares model.
        (w,standardizeF,_) = makeLLS(XtrainBoot,TtrainBoot)
        Wall[:,modeli] = w.flatten()
        StandFall.append(standardizeF)
    
    ######################################################################
    ### Apply all models to test data
    Yall = np.zeros((Xtest.shape[0],nModels))
    for modeli in range(nModels):
        Yall[:,modeli] = useLLS(Wall[:,modeli],StandFall[modeli],Xtest).flatten()
    Ytest = np.mean(Yall,axis=1)
    Ttest = Ttest.flatten()  ## WATCH OUT!! Next line produces wrong scalar without this line!
    RMSEtest = np.sqrt(np.mean((Ytest - Ttest)**2))
    print "Test RMSE is %.4f" % RMSEtest
    ######################################################################
    ### Make plots
    #
    # plot the original data versus sample #
    plt.subplot(4,1,1)
    plt.plot(Xdata[:,1:],'o-')
    plt.ylim(-50000, 70000)
  
    # plot all the models and original data
    plt.subplot(4,1,2)
    plt.plot(Xdata[:,1:],'o-')
    plt.plot(Xtest[:,1:],Yall, '*')
    plt.ylim(-50000, 70000)

    # calculate the confidence levels
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

    # calculate the weight distributions
    nAttributes = Xdata.shape[1]
    lowerW = np.zeros((nAttributes))
    upperW = np.zeros((nAttributes))
    for wi in range(nAttributes):
        wis = Wall[wi,:]
        wis.sort()
        lowerW[wi] = wis[loweri]
        upperW[wi] = wis[upperi]

    printWeights(Wall, lowerW, upperW, nAttributes)

    plt.subplot(4,1,3)
    plt.plot(Xdata[:,1:],'o-')
    plt.fill_between(Xtest[:,1],lower,upper,alpha=0.4)
    plt.ylim(-50000, 70000)

    plt.subplot(4,1,4)
    plt.boxplot(Wall.T)
    plt.xlabel("Attribute")
    plt.ylabel("Distribution of weight values")

    plt.show()
    ######################################################################
