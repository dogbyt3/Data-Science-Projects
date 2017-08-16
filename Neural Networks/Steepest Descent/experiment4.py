import numpy as np
import matplotlib.pyplot as plt
import random
import nnlib
import linearlib

def main():
    # number of samples to create
    n = 20

    # number of repetitions to run Linear Model and NN for
    nReps = 200

    # data set percentages
    train_pct = 0.60
    test_pct = 0.20
    valid_pct = 0.20

    tot_linear_training_errors = []
    tot_linear_testing_errors = []
    tot_avg_linear_training_errors = []
    tot_avg_linear_testing_errors = []

    tot_nn_training_errors = []
    tot_nn_testing_errors = []
    tot_avg_nn_training_errors = []
    tot_avg_nn_testing_errors = []

    for i in range(nReps):

        # create some random independent variable data
        X = np.linspace(0.,20.,n).reshape((n,1))

        # create some random nonlinear dependent variable data
        T = 0.2 + 0.05 * X + 0.4 * np.sin(X) + 0.05 * np.random.normal(size=(n,1))

        # set number of inputs and outputs
        nSamples = X.shape[0]
        nOutputs = T.shape[1]

        # set hidden layers, and learning rates of neural network
        nnReps = 1000
        nHiddens = 10
        rhoh = 0.5
        rhoo = 0.01
        rh = rhoh / (nSamples*nOutputs)
        ro = rhoo / (nSamples*nOutputs)
        weight_dist = 0.1

        # create the neural net and use its distribution of data sets
        # create a neuralNet object
        nn = nnlib.neuralNet(nSamples, nOutputs, nHiddens, rhoh, rhoo, weight_dist)

        # make the Training, Testing, and Validation sets
        nn.makeDataSets(X, T, train_pct, test_pct, valid_pct)

        (testing_set_ind, testing_set_dep, training_set_ind, training_set_dep) = nn.getDataSets()
    
        # make the linear model
        weights = linearlib.makeLLS(training_set_ind, training_set_dep)
            
        # calculate the RMSE for the training set
        training_error = linearlib.calculateRMSE(weights, training_set_ind, training_set_dep)
        tot_linear_training_errors.append(training_error)

        # calculate the RMSE for the testing set
        testing_error = linearlib.calculateRMSE(weights, testing_set_ind, testing_set_dep)
        tot_linear_testing_errors.append(testing_error)

        avg_linear_training_error = np.mean(tot_linear_training_errors)
        avg_linear_testing_error = np.mean(tot_linear_testing_errors)
        tot_avg_linear_training_errors.append(avg_linear_training_error)
        tot_avg_linear_testing_errors.append(avg_linear_testing_error)

        # train and run the Neural Net
        nn.run(nnReps)
        # get the errors from the nnet
        (nn_testing_error, nn_training_error) = nn.get_Errors()
        tot_nn_testing_errors.append(nn_testing_error)
        tot_nn_training_errors.append(nn_training_error)

        avg_nn_training_error = np.mean(tot_nn_training_errors)
        avg_nn_testing_error = np.mean(tot_nn_testing_errors)
        tot_avg_nn_training_errors.append(avg_nn_training_error)
        tot_avg_nn_testing_errors.append(avg_nn_testing_error)


    # plot the avg RMS training and testing errors versus testing set fraction
    plotRMSErrors(nReps, tot_avg_linear_training_errors, tot_avg_linear_testing_errors, tot_avg_nn_training_errors, tot_avg_nn_testing_errors)

def plotRMSErrors(nReps, tot_avg_linear_training_errors, tot_avg_linear_testing_errors, tot_avg_nn_training_errors, tot_avg_nnr_testing_errors):

    plt.clf()
    plt.plot(range(nReps), tot_avg_linear_training_errors, label="linear training rmse")
    plt.plot(range(nReps), tot_avg_linear_testing_errors,  label="linear testing rmse")

    plt.plot(range(nReps), tot_avg_nn_training_errors, label="NN training rmse")
    plt.plot(range(nReps), tot_avg_nnr_testing_errors,  label="NN testing rmse")

    plt.ylabel("RMS Error")
    plt.xlabel("Experimental Iteration")
    plt.legend(('linear training rmse', 'linear testing rmse', 'nn training rmse', 'nn testing rmse')) 
    plt.show()

#prefer to explicitly call my main()
if __name__ == '__main__':
    main()


