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

print "predictions.shape: "+str(np.shape(predictions))
# Plot 1: RMSE for training data versus scg epochs
#plt.figure(1)
#plt.clf()
#nnet.plotError()
#plt.show()


# loop over 5-10 different hidden weight number values using the validation data
hiddenUnits = np.array([1,2,3,4,5,6,7,8,9,10])
plt.figure(1)
plt.clf()

tot_mpg_train_errors = []
tot_mpg_test_errors = []
tot_hp_train_errors = []
tot_hp_test_errors = []
for hidden in hiddenUnits:

    # Construct and train a neural network with different hidden units to approximately 
    # fit Xtrain and Ttrain
    nnet = NeuralNet(Xtrain,Ttrain,hidden,nIterations=100,errorPrecision=1.e-4,weightPrecision=1.e-4)
    Ytrain = nnet.use(Xtrain)
    mpg_Ytrain = Ytrain[0]
    hp_Ytrain = Ytrain[1]
    mpg_Ttrain = Ttrain[0]
    hp_Ttrain = Ttrain[1]
    mpg_error = mpg_Ytrain - mpg_Ttrain
    hp_error = hp_Ytrain - hp_Ttrain
    mpg_rmse = sqrt(np.mean((mpg_error**2).flat))
    hp_rmse = sqrt(np.mean((hp_error**2).flat))
    tot_mpg_train_errors.append(mpg_rmse)
    tot_hp_train_errors.append(hp_rmse)

    # run the nnet on the test data
    Ytest = nnet.use(Xtest)
    mpg_Ytest = Ytest[0]
    hp_Ytest = Ytest[1]
    mpg_Ttest = Ttest[0]
    hp_Ttest = Ttest[1]
    mpg_error = mpg_Ytest - mpg_Ttest
    hp_error = hp_Ytest - hp_Ttest
    mpg_rmse = sqrt(np.mean((mpg_error**2).flat))
    hp_rmse = sqrt(np.mean((hp_error**2).flat))
    tot_mpg_test_errors.append(mpg_rmse)
    tot_hp_test_errors.append(hp_rmse)


plt.subplot(3,1,1)
plt.plot(hiddenUnits, tot_mpg_train_errors, hiddenUnits, tot_hp_train_errors,\
         hiddenUnits, tot_mpg_test_errors, hiddenUnits, tot_hp_test_errors)
plt.legend(("MPG Training","HP Training","MPG Testing","HP Testing"),"lower right")
plt.ylabel("RMS Error")
plt.xlabel("Number Hidden Units")

plt.subplot(3,1,2)
plt.plot(mpg_Ttrain, mpg_Ytrain, mpg_Ttest, mpg_Ytest)
plt.legend(("mpgTraining","mpgTesting"), "lower right")
plt.ylabel("MPG Predicted Value")
plt.xlabel("MPG Actual Value")

plt.subplot(3,1,3)
plt.plot(hp_Ttrain, hp_Ytrain, hp_Ttest, hp_Ytest)
plt.legend(("hpTraining","hpTesting"), "lower right")
plt.ylabel("HP Predicted Value")
plt.xlabel("HP Actual Value")
plt.show()




