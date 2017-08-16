import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import random

class neuralNet(object):
    def __init__(self, nSamples, nOutputs, nHiddens, rhoh, rhoo, weight_dist):
        self.nSamples = nSamples
        self.nOutputs = nOutputs
        self.nHiddens = nHiddens
        self.rhoh = rhoh
        self.rhoo = rhoo
        self.weight_dist = weight_dist
        self.rh = rhoh / (nSamples*nOutputs)
        self.ro = rhoo / (nSamples*nOutputs)
        self.V = weight_dist*2*(np.random.uniform(size=(1+1,nHiddens))-0.5)
        self.W = weight_dist**2*(np.random.uniform(size=(1+nHiddens,nOutputs))-0.5)

    def addOnes(self, A):
        return np.hstack((np.ones((A.shape[0],1)),A))

    def getDataSets(self):
        return (self.testing_set_ind, self.testing_set_dep, self.training_set_ind, self.training_set_dep)

    def makeDataSets(self, original_ind_data, original_dep_data, train_pct, test_pct, valid_pct):
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
        self.training_set_ind = original_ind_data[training_set_indices,:]
        self.valid_set_ind = original_ind_data[validation_set_indices,:]

        # note: for this experiment, make the testing data slightly different than training
        self.testing_set_ind = self.training_set_ind + 0.1*np.random.normal(size=(len(self.training_set_ind),1))

        # create dependent data
        self.training_set_dep = original_dep_data[training_set_indices,:]
        self.valid_set_dep = original_dep_data[validation_set_indices,:]

        # note: for this experiment, make the testing data slightly different than training
        self.testing_set_dep = 0.2 + 0.05 * self.training_set_ind + 0.4 * np.sin(self.testing_set_ind) + 0.05 * np.random.normal(size=(len(self.training_set_dep),1))

        # add bias to the independent sets
        self.training_set_ind1 = self.addOnes(self.training_set_ind)
        self.testing_set_ind1 = self.addOnes(self.testing_set_ind)
        self.valid_set_ind1 = self.addOnes(self.valid_set_ind)

    def train(self, nReps):

        # collect testing and validation errors for plotting
        errorTrace = np.zeros((nReps,2))
        errorTrace[:] = np.nan

        X = self.training_set_ind
        T = self.training_set_dep
        Ttest = self.testing_set_dep
        Xtest = self.testing_set_ind
        X1 = self.training_set_ind1
        Xtest1 = self.testing_set_ind1

        Xvalid = self.valid_set_ind
        Xvalid1 = self.valid_set_ind1
        Tvalid = self.valid_set_dep

        for reps in range(nReps):
            # Forward pass on training data
            Z = np.tanh(np.dot( X1, self.V ))
            Z1 = self.addOnes(Z)
            Y = np.dot( Z1, self.W )

            # Error in output
            error = Y - T

            # Backward pass - the backpropagation and weight update steps
            self.V = self.V - self.rh * np.dot( X1.T, np.dot( error, self.W[1:,:].T) * (1-Z**2))
            self.W = self.W - self.ro * np.dot( Z1.T, error)

            # create the test model
            Ytest = np.dot(self.addOnes(np.tanh(np.dot(Xtest1,self.V))), self.W)

            # update the testing error
            errorTrace[reps,0] = sqrt(np.mean(((Ytest-Ttest)**2).flat))

            # create the validation model
            Yvalid = np.dot(self.addOnes(np.tanh(np.dot(Xvalid1,self.V))), self.W)

            # update the validation error
            errorTrace[reps,1] = sqrt(np.mean(((Yvalid-Tvalid)**2).flat))

            # Every so often update the graphs
            if reps % (nReps/50) == 0:
                plt.ion()
                plt.figure(1)
                plt.clf()
                plt.subplot(3,1,1)
                plt.ylabel("Error")
                plt.xlabel('NN Repetitions')
                plt.plot(np.arange(nReps),errorTrace)
                plt.legend(('Testing Error', 'Validation Error'), 'upper right')

                plt.subplot(3,1,2)
                plt.plot(X,T,'o-',Xtest,Ttest,'o-',Xtest,Ytest,'o-')
                plt.ylabel("Models")
                plt.legend(('Training','Testing','Model'),'lower right')
        
                plt.subplot(3,1,3)
                plt.plot(X,Z)
                plt.ylabel("Hidden Units = "+str(self.nHiddens))
                plt.draw()


        x = raw_input("Press Enter to continue...")

        self.testing_error = errorTrace[reps, 0]
        self.valid_error = errorTrace[reps, 1]

        
    def get_FinalErrors(self):
        return (self.testing_error, self.valid_error)

    def get_Errors(self):
        return (self.testing_error, self.training_error)

    def run(self, nReps):
        # collect training and testing errors for plotting
        errorTrace = np.zeros((nReps,2))
        errorTrace[:] = np.nan

        X = self.training_set_ind
        T = self.training_set_dep
        Ttest = self.testing_set_dep
        Xtest = self.testing_set_ind
        X1 = self.training_set_ind1
        Xtest1 = self.testing_set_ind1

        for reps in range(nReps):
            # Forward pass on training data
            Z = np.tanh(np.dot( X1, self.V ))
            Z1 = self.addOnes(Z)
            Y = np.dot( Z1, self.W )

            # Error in output
            error = Y - T

            # Backward pass - the backpropagation and weight update steps
            self.V = self.V - self.rh * np.dot( X1.T, np.dot( error, self.W[1:,:].T) * (1-Z**2))
            self.W = self.W - self.ro * np.dot( Z1.T, error)

            # error traces for plotting
            # update the training error
            errorTrace[reps,0] = sqrt(np.mean((error**2).flat))
            self.training_error = errorTrace[reps,0]

            # create the model
            Ytest = np.dot(self.addOnes(np.tanh(np.dot(Xtest1,self.V))), self.W)

            # update the testing error
            errorTrace[reps,1] = sqrt(np.mean(((Ytest-Ttest)**2).flat))
            self.testing_error = errorTrace[reps,1]

