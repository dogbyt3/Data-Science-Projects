import numpy as np
from math import sqrt
import matplotlib.pyplot as plt

# Make some training data
n = 20
X = np.linspace(0.,20.0,n).reshape((n,1))
T = 0.2 + 0.05 * X + 0.4 * np.sin(X) + 0.05 * np.random.normal(size=(n,1))

# Make some testing data
Xtest = X + 0.1*np.random.normal(size=(n,1))
Ttest = 0.2 + 0.05 * X + 0.4 * np.sin(Xtest) + 0.1 * np.random.normal(size=(n,1))

nSamples = X.shape[0]
nOutputs = T.shape[1]

# Set parameters of neural network
nHiddens = 10
rhoh = 0.5
rhoo = 0.01
rh = rhoh / (nSamples*nOutputs)
ro = rhoo / (nSamples*nOutputs)

# Initialize weights to uniformly distributed values between small normally-distributed between -0.1 and 0.1
V = 0.1*2*(np.random.uniform(size=(1+1,nHiddens))-0.5)
W = 0.1*2*(np.random.uniform(size=(1+nHiddens,nOutputs))-0.5)

# Add constant column of 1's
def addOnes(A):
    return np.hstack((np.ones((A.shape[0],1)),A))
X1 = addOnes(X)
Xtest1 = addOnes(Xtest)

# Take nReps steepest descent steps in gradient descent search in mean-squared-error function
nReps = 200000
# collect training and testing errors for plotting
errorTrace = np.zeros((nReps,2))
errorTrace[:] = np.nan
for reps in range(nReps):

    # Forward pass on training data
    Z = np.tanh(np.dot( X1, V ))
    Z1 = addOnes(Z)
    Y = np.dot( Z1, W )

    # Error in output
    error = Y - T

    # Backward pass - the backpropagation and weight update steps
    V = V - rh * np.dot( X1.T, np.dot( error, W[1:,:].T) * (1-Z**2))
    W = W - ro * np.dot( Z1.T, error)

    # error traces for plotting
    errorTrace[reps,0] = sqrt(np.mean((error**2).flat))
    Ytest = np.dot(addOnes(np.tanh(np.dot(Xtest1,V))), W)
    errorTrace[reps,1] = sqrt(np.mean(((Ytest-Ttest)**2).flat))

    # Every so often update the graphs
    if reps % (nReps/500) == 0:
        plt.ion()
        plt.figure(1)
        plt.clf()
        plt.subplot(3,1,1)
        plt.plot(np.arange(nReps),errorTrace)

        plt.subplot(3,1,2)
        plt.plot(X,T,'o-',Xtest,Ttest,'o-',Xtest,Ytest,'o-')
        plt.legend(('Training','Testing','Model'),'lower right')
        
        plt.subplot(3,1,3)
        plt.plot(X,Z)
        plt.draw()