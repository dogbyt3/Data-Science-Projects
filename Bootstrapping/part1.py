"""
* Author:  Michael Barber
* File:    part1.py


"""

import numpy as np
import matplotlib.pyplot as plt
import random
from linear_lib_p1 import * # for linear model functions

def main():
    
    ######################################################################
    ### Parameters
    trainingFraction = 0.5 # percentage of training/test fractions
    nModels = 20           # number of models
    confidenceLevel = 90   # confidence levels in percent
    nData = 20             # number of samples
    nTest = 50             # number of test samples
    ######################################################################

    # X is data between 0 and 3, and 6 and 10
    Xdata = np.hstack((np.linspace(0, 3, num=nData),
               np.linspace(6, 10, num=nData))).reshape((2*nData,1))
    Tdata = -1 + 0.4 * Xdata + 2*np.random.normal(size=(2*nData,1))


    print("Modify CI: ")
    print("CalculateConfidenceLevels(X, T, 0.5, 20, 90, 20, 50)")
    calculateConfidenceLevels(Xdata, Tdata, trainingFraction, nModels, confidenceLevel,
                              nData, nTest)

    print("CalculateConfidenceLevels(X, T, 0.5, 20, 100, 20, 50)")
    calculateConfidenceLevels(Xdata, Tdata, trainingFraction, nModels, 99,
                              nData, nTest)

    print("CalculateConfidenceLevels(X, T, 0.5, 20, 10, 20, 50)")
    calculateConfidenceLevels(Xdata, Tdata, trainingFraction, nModels, 10,
                              nData, nTest)


    
#prefer to explicitly call my main()
if __name__ == '__main__':
    main()
