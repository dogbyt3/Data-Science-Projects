import numpy as np
import matplotlib.pyplot as plt
from NeuralNet import *

tot_train_errors = []
avg_training_error = []
tot_avg_train_errors = []
tot_test_errors = []
avg_test_error = []
tot_avg_test_errors = []

for hidden in range(1, 100):

    # Make some training data
    n = 40
    X = np.linspace(0.,20.0,n).reshape((n,1))
    T = -1 + 0.05 * X + 0.4 * np.sin(X) + 0.03 * np.random.normal(size=(n,1))

    # Make some testing data
    Xtest = X + 0.1*np.random.normal(size=(n,1))
    Ttest = -1 + 0.05 * X + 0.4 * np.sin(Xtest) + 0.03 * \
                                  np.random.normal(size=(n,1))

    nSamples = X.shape[0]
    nOutputs = T.shape[1]

    # Construct and train a neural network with different hidden units to approximately 
    # fit Xtrain and Ttrain
    nnet = NeuralNet(X,T,hidden,nIterations=100,errorPrecision=1.e-4,weightPrecision=1.e-4)
    Ytrain = nnet.use(X)
    error = Ytrain - T
    rmse = sqrt(np.mean((error**2).flat))
    tot_train_errors.append(rmse)

    # use the network to calculate testing rmse
    Ytest = nnet.use(Xtest)
    error = Ytest - Ttest
    rmse = sqrt(np.mean((error**2).flat))
    tot_test_errors.append(rmse)

# Plot the train and test RMSE versus the number of hidden units. for 1-20, or more hidden units
#####
##### run nnet 20 or so times with different hidden unit amounts, save errors
#####
plt.figure(1)
plt.clf()
plt.plot(range(1,100), tot_train_errors, range(1,100), tot_test_errors)
plt.legend(("Training RMSE", "Testing RMSE"),"lower right")

#plt.plot(range(1,100), tot_test_errors, label="testing rmse")
plt.ylabel("RMS Error")
plt.xlabel("Number Hidden Units")
plt.show()


