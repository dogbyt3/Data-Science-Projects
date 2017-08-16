import numpy as np
import matplotlib.pyplot as plt
import random
from NeuralNet import *

# 1. mpg:           continuous
# 2. cylinders:     multi-valued discrete
# 3. displacement:  continuous
# 4. horsepower:    continuous
# 5. weight:        continuous
# 6. acceleration:  continuous
# 7. model year:    multi-valued discrete
# 8. origin:        multi-valued discrete
# 9. car name:      string (unique for each instance)

names =  ['mpg','cylinders','displacement','horsepower','weight',
          'acceleration','year','origin']

def plotPredictions(targets, predictions):
    plt.subplot(2, 1, 1)
    plt.plot(targets[:, 0], 'o-')
    plt.plot(predictions[:, 0], 'o-')
    plt.legend(("Testing MPG", "Model"), 'upper right')

    plt.subplot(2, 1, 2)
    plt.plot(targets[:, 1], 'x-')
    plt.plot(predictions[:, 1], 'x-')
    plt.legend(("Testing HP", "Model"), 'upper right')

def missingIsNan(s):
    if s == '?':
        return np.nan
    else:
        return float(s)

def partitionSet(data):

    original_ind_data = np.concatenate(((data[:,1:3]), (data[:,4:])), axis=1)
    original_dep_data = np.concatenate(((data[:,0:3]), (data[:,3:4])), axis=1)

    train_pct = 0.60
    test_pct = 0.20
    valid_pct = 0.20

    # indices for all data rows
    nrow = np.shape(original_ind_data)[0]
    all_indices = xrange(nrow)

    training_size = int(round(nrow*train_pct))
    testing_size = int(round(nrow*test_pct))
    valid_size = int(round(nrow*valid_pct))

    # number of training samples
    training_set_size = int(round(nrow*train_pct))

    remaining_set_size = nrow - training_set_size

    # row indices for training samples
    training_set_indices = list(set(random.sample(all_indices,training_set_size))) 

    # row indices of remaining samples
    remaining_indices = list(set(all_indices).difference(set(training_set_indices)))
 
    # adjusted testing set sample percentage for the remainder data
    adj_pct = ((test_pct*nrow)/(nrow-training_size))

    # row indices of testing samples
    testing_set_indices = list(set(random.sample(remaining_indices,\
                                   int(round(len(remaining_indices)*adj_pct)))))                      

    # remaining belong to validation set
    validation_set_indices = list(set(remaining_indices).difference(set(testing_set_indices)))

    # create independent data
    training_set_ind = original_ind_data[training_set_indices,:]
    testing_set_ind = original_ind_data[testing_set_indices,:]
    valid_set_ind = original_ind_data[validation_set_indices,:]

    # create dependent data
    training_set_dep = original_dep_data[training_set_indices,:]
    testing_set_dep = original_dep_data[testing_set_indices,:]
    valid_set_dep = original_dep_data[validation_set_indices,:]

    return(training_set_ind, training_set_dep, testing_set_ind, testing_set_dep,\
           valid_set_ind, valid_set_dep)         

dataOriginal = np.loadtxt('data/auto-mpg.data',usecols=range(8),
                          converters={3: missingIsNan})

notNans = np.isnan(dataOriginal) == False
goodRowsMask = notNans.all(axis=1)
data = dataOriginal[goodRowsMask,:]
(Xtrain, Ttrain, Xtest, Ttest, Xvalid, Tvalid) = partitionSet(data)

nnet = NeuralNet(Xtrain,Ttrain,10,nIterations=100,errorPrecision=1.e-4,weightPrecision=1.e-4)
predictions = nnet.use(Xtrain)

# Plot 1: RMSE for training data versus scg epochs
#plt.figure(1)
#plt.clf()
#nnet.plotError()
#plt.show()


# loop over 5-10 different hidden weight number values using the validation data
hiddenUnits = np.array([1,2,3,4,5])
#hiddenUnits = np.array([1,2,3,4,5,6,7,8,9,10])
#plt.figure(1)
#plt.clf()

errors = np.zeros((len(hiddenUnits), 2))
errors[:] = np.nan
i = 0
for hidden in hiddenUnits:

    # Construct and train a neural network with different hidden units to approximately 
    # fit Xtrain and Ttrain
    nnet = NeuralNet(Xtrain,Ttrain,hidden,nIterations=1000,errorPrecision=0,weightPrecision=0)

    # run the network on training data set
    stdTrain = nnet.standardize(Xtrain)
    TrainPredictions = nnet.use(stdTrain)

    # now run it on validation data set
    stdValid = nnet.standardize(Xvalid)
    ValidPredictions = nnet.use(stdValid)
    
    errors[i, 0] = sqrt(np.mean(((TrainPredictions - Ttrain)**2).flat))
    errors[i, 1] = sqrt(np.mean(((ValidPredictions - Tvalid)**2).flat))
    i = i + 1
    
    #plt.figure(i)
    #plt.clf()
    #plotPredictions(Tvalid, ValidPredictions)

#plt.show()

# now determine the best hidden unit value to use from the errors
validErrors=errors[:,1]
low=999
index=0
for j in range(len(validErrors)):
    if (validErrors[j] < low):
        low = validErrors[j]
        index = j
    #get the best hidden unit value
    best = hiddenUnits[index]


# use the network with the best number of 

net2 = NeuralNet(Xtrain, Ttrain, best, weightPrecision=0, errorPrecision=0, nIterations=1000)
#use the test data in the neural network
stdTest = net2.standardize(Xtest)
TestPredictions = net2.use(stdTest)
#plot the predictions versus actuals for test data

plt.figure(1)
plt.clf()
plotPredictions(Ttest, TestPredictions)
#plotPredVsAct(Ttest, predictionsTest, 0, 1, 'Testing MPG', 'Model MPG')
#plotPredVsAct(Ttest, predictionsTest, 1, 2, 'Testing Horsepower', 'Model Horsepower')
	
#plot the RMSE values
plt.ion()
plt.figure(2)
plt.clf()
plt.plot(hiddenUnits, errors)
plt.legend(("HiddenUnits", "Errors"), 'upper right')
plt.show()


