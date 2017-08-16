"""
* Author:  Michael Barber
* File:    wine_quality.py
* Purpose: This module reads in data file(s) of wine characteristics obtained
           from <http://archive.ics.uci.edu/ml/datasets/Wine+Quality>, 
           standardizes the independent variables, 

"""

import numpy as np
import matplotlib.pyplot as plt
import csv
import sys
import random
from linear_lib import *

def main():
    # Red wine data file
    red_file = 'data/winequality-red.csv'
    # White wine data file
    white_file = 'data/winequality-white.csv'

    # open the red wine data file
    original_data = open_file(red_file)

    # prepare the data for the linear model w./ 80% distribution
    (training_set_ind, training_set_dep, \
            testing_set_ind, testing_set_dep, \
            dependent_name, independent_names) = prepareDataForModel(original_data, 0.95)

    # make the linear model
    weights = makeLLS(training_set_ind, training_set_dep)

    # print the weights of the training data
    printWeights(weights, independent_names)

    # use the linear model of the training set to predict
    # the quality of the testing set
    prediction = useLLS(weights, testing_set_ind)

    # plot the prediction versus observed
    plotPredictions(prediction, testing_set_dep)

    # calculate the root-mean-square errors (RMSE)
    error = calculateRMSE(weights, testing_set_ind, testing_set_dep)

    print("error: "+str(error))


#prefer to explicitly call my main()
if __name__ == '__main__':
    main()
