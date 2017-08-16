"""
* Author:  Michael Barber
* File:    part3.py


"""

import numpy as np
import matplotlib.pyplot as plt
import random
from linear_lib import * # for linear model functions

def main():
    
    ######################################################################
    ### Parameters
    trainingFraction = 0.75 # percentage of training/test fractions
    nModels = 1000          # number of models
    confidenceLevel = 90    # confidence levels in percent
    nData = 20              # number of samples
    nTest = 50              # number of test samples
    ######################################################################

    # open the uci repository machine data file
    ######################################################################
    #   1. vendor name: 30 
    #  (adviser, amdahl,apollo, basf, bti, burroughs, c.r.d, cambex, cdc, dec, 
    #   dg, formation, four-phase, gould, honeywell, hp, ibm, ipl, magnuson, 
    #   microdata, nas, ncr, nixdorf, perkin-elmer, prime, siemens, sperry, 
    #   sratus, wang)
    #   2. Model Name: many unique symbols
    #    3. MYCT: machine cycle time in nanoseconds (integer)
    #    4. MMIN: minimum main memory in kilobytes (integer)
    #    5. MMAX: maximum main memory in kilobytes (integer)
    #    6. CACH: cache memory in kilobytes (integer)
    #    7. CHMIN: minimum channels in units (integer)
    #    8. CHMAX: maximum channels in units (integer)
    #    9. PRP: published relative performance (integer)
    #    10. ERP: estimated relative performance from the original article (integer)
    ######################################################################

    original_data = open_file("./data/machine.data")
    
    # create training and test data sets
    (Xdata, Tdata, Ttrain, Xtrain, Ttest, Xtest) = \
    create_sets(original_data, trainingFraction)

#    print("Xdata: "+str(Xdata))
#    print("Xdata.shape(): ["+str((np.shape(Xdata)[0]))+"]["+str(np.shape(Xdata)[1])+"]")
#    print("Tdata.shape(): ["+str(np.shape(Tdata)[0])+"]["+str(np.shape(Tdata)[1])+"]")

#    print("Ttrain.shape(): ["+str((np.shape(Ttrain)[0]))+"]["+str(np.shape(Ttrain)[1])+"]")
#    print("Xtrain.shape(): ["+str((np.shape(Xtrain)[0]))+"]["+str(np.shape(Xtrain)[1])+"]")

#    print("Tdata: "+str(Tdata))

    calculateConfidenceLevels(Xdata, Tdata, Ttrain, Xtrain, Ttest, Xtest, 
                              trainingFraction, nModels, confidenceLevel,
                              nData, nTest)
    
#prefer to explicitly call my main()
if __name__ == '__main__':
    main()
