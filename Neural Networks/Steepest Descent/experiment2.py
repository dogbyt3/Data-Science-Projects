import numpy as np
import nnlib
import random

def main():
    # number of samples to create
    n = 20

    # number of repititions to run NN for
    nReps = 200000

    # data set percentages
    train_pct = 0.60
    test_pct = 0.20
    valid_pct = 0.20

    # create some random independent variable data
    X = np.linspace(0.,20.,n).reshape((n,1))

    # create some random nonlinear dependent variable data
    T = 0.2 + 0.05 * X + 0.4 * np.sin(X) + 0.05 * np.random.normal(size=(n,1))

    # set number of inputs and outputs
    nSamples = X.shape[0]
    nOutputs = T.shape[1]

    # set hidden layers, and learning rates of neural network
    nHiddens = 10
    rhoh = 0.5                          
    rhoo = 0.01
    rh = rhoh / (nSamples*nOutputs)
    ro = rhoo / (nSamples*nOutputs)
    weight_dist = 0.1

    # create a neuralNet object
    nn = nnlib.neuralNet(nSamples, nOutputs, nHiddens, rhoh, rhoo, weight_dist)

    # make the Training, Testing, and Validation sets
    nn.makeDataSets(X, T, train_pct, test_pct, valid_pct)

    # run the net and plot the results
    nn.run(nReps)


#prefer to explicitly call my main()
if __name__ == '__main__':
    main()


