import numpy as np
import matplotlib.pyplot as plt
import random
import nnlib

def main():
    # number of samples to create
    n = 20

    # number of repititions to run NN for
    nReps = 200

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
    hidden_array = np.array([1, 5, 10, 25, 50])
    rhoh = 0.5
    rhoo = 0.01
    rh = rhoh / (nSamples*nOutputs)
    ro = rhoo / (nSamples*nOutputs)
    weight_dist = 0.1

    tot_test_errors = []
    tot_valid_errors = []

    # loop through the array of nHiddens and create a nn from it
    for i in range(np.size(hidden_array)):
        # create a neuralNet object
        nn = nnlib.neuralNet(nSamples, nOutputs, hidden_array[i], rhoh, rhoo, weight_dist)

        # make the Training, Testing, and Validation sets
        nn.makeDataSets(X, T, train_pct, test_pct, valid_pct)

        # run the nnet and plot the results
        nn.train(nReps)

        # get the errors from the nnet
        (testing_error, validation_error) = nn.get_FinalErrors()
        tot_test_errors.append(testing_error)
        tot_valid_errors.append(validation_error)

    # plot the avg RMS training and testing errors versus testing set fraction
    plotRMSErrors(hidden_array, tot_test_errors, tot_valid_errors)


def plotRMSErrors(hidden_array, tot_testing_errors, tot_valid_errors):
    plt.clf()
    plt.plot(hidden_array, tot_testing_errors, label="testing rmse")
    plt.plot(hidden_array, tot_valid_errors,   label="validation rmse")
    plt.ylabel("RMS Error")
    plt.xlabel("Hidden Units")
    plt.legend(('testing rmse', 'validation rmse')) 
    plt.show()

#prefer to explicitly call my main()
if __name__ == '__main__':
    main()


