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
    red_data = open_file(red_file)

    # open the white wine data file
    white_data = open_file(white_file)

    # pass in the white wine and use it as the training set
    (white_training_set_ind, white_training_set_dep, \
         white_testing_set_ind, white_testing_set_dep, \
         white_dependent_name, white_independent_names) = \
         prepareDataForModel(white_data, 1)
    
    
    # using the same prepareDataForModel, pass in the red wine
    # and use 100% of it as the testing set
    (red_training_set_ind, red_training_set_dep, \
         red_testing_set_ind, red_testing_set_dep, \
         red_dependent_name, red_independent_names) = \
         prepareDataForModel(red_data, 1)


    # make the linear model
    weights = makeLLS(white_training_set_ind, white_training_set_dep)

    # print the weights of the training data
    printWeights(weights, white_independent_names)

    # use the linear model of the training set to predict
    # the quality of the testing set
    prediction = useLLS(weights, red_training_set_ind)

    # plot the prediction versus observed
    plotPredictions(prediction, red_training_set_dep)

    # calculate the root-mean-square errors (RMSE)
    error = calculateRMSE(weights, red_training_set_ind, red_training_set_dep)

    print("error: "+str(error))


#prefer to explicitly call my main()
if __name__ == '__main__':
    main()
