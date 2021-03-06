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

    testing_set_fractions = np.array([.05, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95])
    tot_avg_training_errors = []
    tot_avg_testing_errors = []
 
   # for each of the testing & training set sizes
    for i in range(np.size(testing_set_fractions)):
        avg_training_error = 0
        avg_training_prediction = 0
        avg_testing_error = 0
        avg_testing_prediction = 0
        tot_training_errors = []
        tot_training_predictions = []
        tot_testing_errors = []
        tot_testing_predictions = []

        for j in range(99):
 
            # prepare the data for the linear model by:
            #    1. divide data into training and testing partitions
            #    2. standardize the training independent variables
            #    3. standardize the testing independent variables using
            #       means and std_dev of training set.
    
            # pass in the white wine and use it as the training set
            (white_training_set_ind, white_training_set_dep, \
                 white_testing_set_ind, white_testing_set_dep, \
                 white_dependent_name, white_independent_names) = \
                 prepareDataForModel(white_data, testing_set_fractions[i])
    
    
            # using the same prepareDataForModel, pass in the red wine
            # and use 100% of it as the testing set
            (red_training_set_ind, red_training_set_dep, \
                 red_testing_set_ind, red_testing_set_dep, \
                 red_dependent_name, red_independent_names) = \
                 prepareDataForModel(red_data, testing_set_fractions[i])
    

            # make the linear model
            weights = makeLLS(white_training_set_ind, white_training_set_dep)
            
            # calculate the RMSE for the training set
            training_error = calculateRMSE(weights, white_training_set_ind, \
                                           white_training_set_dep)
            tot_training_errors.append(training_error)

            # calculate the RMSE for the testing set
            testing_error = calculateRMSE(weights, red_training_set_ind, \
                                          red_training_set_dep)
            tot_testing_errors.append(testing_error)

            # use the linear model of the training set to predict
            # the quality of the training set
            training_prediction = useLLS(weights, white_training_set_ind)
            tot_training_predictions.append(training_prediction)


            # use the linear model of the training set to predict
            # the quality of the testing set
            testing_prediction = useLLS(weights, red_testing_set_ind)
            tot_testing_predictions.append(testing_prediction)


        avg_training_error = np.mean(tot_training_errors)
        avg_testing_error = np.mean(tot_testing_errors)
        tot_avg_training_errors.append(avg_training_error)
        tot_avg_testing_errors.append(avg_testing_error)

        print("Test["+str(testing_set_fractions[i])+\
              "] AvgTrError: "+ str(avg_training_error)+\
              " AvgTstError: "+ str(avg_testing_error))

        

    # plot the avg RMS training and testing errors versus testing set fraction
    plotRMSErrors(testing_set_fractions, tot_avg_training_errors, \
                  tot_avg_testing_errors)


def plotRMSErrors(testing_set_fractions, tot_avg_training_errors, \
                  tot_avg_testing_errors):

    plt.clf()
    plt.plot(testing_set_fractions, tot_avg_training_errors, \
             label="training rmse")
    plt.plot(testing_set_fractions, tot_avg_testing_errors, \
             label="testing rmse")
    plt.ylabel("RMS Error")
    plt.xlabel("Set Fraction")
    plt.legend(('training rmse', 'testing rmse')) 
    plt.show()



#prefer to explicitly call my main()
if __name__ == '__main__':
    main()
