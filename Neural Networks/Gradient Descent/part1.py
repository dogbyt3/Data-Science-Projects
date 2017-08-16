import numpy as np
import matplotlib.pyplot as plt
from NeuralNet import *

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

# Construct and train a neural network with 10 hidden units to approximately 
# fit Xtrain and Ttrain
nnet = NeuralNet(X,T,10,nIterations=1000,errorPrecision=0,weightPrecision=0)

# run the network to get the hidden units
Xhidden = np.linspace(0.,20.0,n).reshape((n,1))
stdXhidden = nnet.standardize(Xhidden)
hiddenUnits = nnet.use(stdXhidden, allOutputs=True)[0]

# run the nework on the training set
stdTrain = nnet.standardize(X)
TrainPredictions = nnet.use(stdTrain)

# run the network on the test set
stdTest = nnet.standardize(Xtest)
TestPredictions = nnet.use(stdTest)

# Plot 1: RMSE for training data versus scg epochs
plt.figure(1)
plt.clf()
plt.subplot(5,1,1)
nnet.plotError()

# Plot 2: Output of hidden units for range of x values from 0 to 20.0.
#####
#####
plt.subplot(5,1,2)
plt.plot(hiddenUnits)
plt.ylabel("Hidden Unit Output")

# Plot 3: Output of training data target with output of neural network given training data input
#####
##### plot T versus predictions from nnet
#####
plt.subplot(5,1,3)
plt.plot(X, T, 'x', X, TrainPredictions, 'x')
plt.legend(("Training", "Model"), "lower right")

# Plot 4: Output of testing data target with output of neural network given testing data input.
#####
##### plot Ttest versus predictions from nnet using Xtest as nnet input
#####
plt.subplot(5,1,4)
plt.plot(Xtest, Ttest, 'x', Xtest, TestPredictions, 'x')
plt.legend(("Testing", "Model"), "lower right")

# Plot 5: Diagram of neural network weights. Use code to be supplied soon.
#####
#####
plt.subplot(5,1,5)
nnet.draw()
plt.show()

