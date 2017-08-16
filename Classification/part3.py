import numpy as np
import matplotlib.pyplot as plt
from ClassifierLib import *

D = 1
f = open("parkinsons.data","r")
header = f.readline()
names = header.strip().split(',')[1:]

data = np.loadtxt(f ,delimiter=',', usecols=1+np.arange(23))

targetColumn = names.index("status")
XColumns = range(23)
XColumns.remove(targetColumn)
X = data[:,XColumns]
T = data[:,targetColumn,np.newaxis]
names.remove("status")

trainf = 0.8
healthyI,_ = np.where(T == 0)
parkI,_ = np.where(T == 1)
healthyI = np.random.permutation(healthyI)
parkI = np.random.permutation(parkI)

nHealthy = len(healthyI)
nParks = len(parkI)

n = round(trainf*len(healthyI))
rows = healthyI[:n]
Xtrain = X[rows,:]
Ttrain = T[rows,:]
rows = healthyI[n:]
Xtest =  X[rows,:]
Ttest =  T[rows,:]
n = round(trainf*len(parkI))
rows = parkI[:n]
Xtrain = np.vstack((Xtrain, X[rows,:]))
Ttrain = np.vstack((Ttrain, T[rows,:]))
rows = parkI[n:]
Xtest = np.vstack((Xtest, X[rows,:]))
Ttest = np.vstack((Ttest, T[rows,:]))

standardize,unstandardize = makeStandardizeF(Xtrain)
Xtrains = standardize(Xtrain)
Xtests = standardize(Xtest)

Ttr = Ttrain.ravel()==0
mu1 = np.mean(Xtrains[Ttr,:],axis=0)
cov1 = np.cov(Xtrains[Ttr,:].T)
Ttr = Ttrain.ravel()==1
mu2 = np.mean(Xtrains[Ttr,:],axis=0)
cov2 = np.cov(Xtrains[Ttr,:].T)

# Form the QDA discriminant functions.
prior1 = float(nHealthy)/(nHealthy+nParks)
prior2 = float(nParks)/(nHealthy+nParks)

d1 = discQDA(Xtrains,standardize,mu1,cov1,prior1)
d2 = discQDA(Xtrains,standardize,mu2,cov2,prior2)
QDApredictedTrain = np.argmax(np.vstack((d1,d2)),axis=0)

d1t = discQDA(Xtests,standardize,mu1,cov1,prior1)
d2t = discQDA(Xtests,standardize,mu2,cov2,prior2)
QDApredictedTest = np.argmax(np.vstack((d1t,d2t)),axis=0)

# create LDA discriminant functions
ld1 = discLDA(Xtrains,standardize,mu1,cov1,prior1)
ld2 = discLDA(Xtrains,standardize,mu2,cov2,prior2)
LDApredictedTrain = np.argmax(np.vstack((ld1,ld2)),axis=0)

# apply LDA discr to testing data
ld1t = discLDA(Xtests,standardize,mu1,cov1,prior1)
ld2t = discLDA(Xtests,standardize,mu2,cov2,prior2)
LDApredictedTest = np.argmax(np.vstack((ld1t,ld2t)),axis=0)

# run Logistic Regression
TtrainI = makeIndicatorVars(Ttrain)
TtestI = makeIndicatorVars(Ttest)

beta = np.zeros((Xtrains.shape[1],TtrainI.shape[1]-1))
alpha = 0.0001
for step in range(1000) :
  gs = g(Xtrains,beta)
  beta = beta + alpha * np.dot(Xtrains.T, TtrainI[:,:-1] - gs[:,:-1])
  likelihoodPerSample = np.exp( np.sum(TtrainI * np.log(gs)) / Xtrains.shape[0])

logregOutput = g(Xtrains,beta)
LRpredictedTrain = np.argmax(logregOutput,axis=1)
logregOutput = g(Xtests,beta)
LRpredictedTest = np.argmax(logregOutput,axis=1)

######## print out the predicted percentages of the three methods
print "QDA Percent correct: Train[",Ttrain.shape,"]",percentCorrect(QDApredictedTrain,Ttrain),"Test[",Ttest.shape,"]",percentCorrect(QDApredictedTest,Ttest)
print "LDA Percent correct: Train[",Ttrain.shape,"]",percentCorrect(LDApredictedTrain,Ttrain),"Test[",Ttest.shape,"]",percentCorrect(LDApredictedTest,Ttest)
print "LR Percent correct: Train[",Ttrain.shape,"]",percentCorrect(LRpredictedTrain,Ttrain),"Test[",Ttest.shape,"]",percentCorrect(LRpredictedTest,Ttest)


###############################################################################
########## QDA Plots
###############################################################################

############# QDA Plot 1
#
# the training data, as its x value versus the class (1, 2 or 3)
#
plt.figure(1)
plt.clf()
plt.subplot(6,1,1)
Ttr = Ttrain.ravel()==0
X1=Xtrains[Ttr,:]
T1=Ttrain[Ttr,:]
Ttr = Ttrain.ravel()==1
X2=Xtrains[Ttr,:]
T2=Ttrain[Ttr,:]

plt.plot(X1, T1, 'bo')
plt.plot(X2, T2, 'ro')
plt.ylim([-0.5,1.5])
plt.ylabel("Train Class")

############# QDA Plot 2
#
# the three curves for p(x|Class=k) for k=1,2.
#
probs = np.exp(np.vstack((d1t-np.log(prior1),d2t-np.log(prior2))).T \
                          - 0.5*D*np.log(2*np.pi))

plt.subplot(6,1,2)
plt.plot(Xtests[:,0],probs[:,0],"r")
plt.plot(Xtests[:,0],probs[:,1],"b")
plt.ylabel("QDA"+" $p(x|class=k)$")  

############# QDA Plot 3
#
# the curve for p(x) for the test data
# p(x)=sum(k=1,K)[(p(x|C=k)*p(C=k))] = sum(k=1,K)[p(x|C=k)*1/2]
probs2 = np.array(((probs)/2.0),dtype=np.float32)
px = np.sum(probs2,axis=1)[:,np.newaxis]
makePlot3(Xtests, px, "QDA")

############# QDA Plot 4
#
# the two curves for p(C=k|x) for k = 1, and 2, for the test data
# p(C=k|x) = (p(x|C=k) * p(C=k))/p(x) = ((probs)(1/2))/(probs2)
#
probs3 = np.array(((probs/2.0)/px),dtype=np.float32)
makePlot4(Xtests, probs3, "QDA")

############# QDA Plot 5
#
# the two discriminant functions for the test data
#
plt.subplot(6,1,5)
plt.plot(Xtests[:,0],np.vstack((d1t,d2t)).T)
plt.ylabel("QDA"+" $\delta_{k}$")

############# QDA Plot 6
#
# the class predicted by the classifier for the test data
#
predictedTest = np.argmax(np.vstack((d1t,d2t)),axis=0)[:,np.newaxis]
# increment returned indices
predictedTest = predictedTest+1
plt.subplot(6,1,6)
plt.plot(Xtests[:,0], predictedTest, '_')
plt.ylabel("QDA"+" predicted class")
plt.ylim([-0.5,2.5])
plt.show()
predictedTrain = np.argmax(np.vstack((d1,d2)),axis=0)

###############################################################################
########## LDA Plots
###############################################################################

############# LDA Plot 1
#
# the training data, as its x value versus the class (1, 2 or 3)
#
plt.figure(1)
plt.clf()
plt.subplot(6,1,1)
Ttr = Ttrain.ravel()==0
X1=Xtrains[Ttr,:]
T1=Ttrain[Ttr,:]
Ttr = Ttrain.ravel()==1
X2=Xtrains[Ttr,:]
T2=Ttrain[Ttr,:]

plt.plot(X1, T1, 'bo')
plt.plot(X2, T2, 'ro')
plt.ylim([-0.5,1.5])
plt.ylabel("Train Class")

############# LDA Plot 2
#
# the three curves for p(x|Class=k) for k=1,2,3, for x values in a set of test 
# data generated by x = np.linspace(0,4,100), where p(x|C=k) is calculated 
# using means and standard deviations for each class calculated from the 
# training data
#
prob1 = normald(standardize(Xtests), mu1, cov1)
prob2 = normald(standardize(Xtests), mu2, cov2)
probs = np.hstack((prob1,prob2))
makePlot2(Xtests, probs, "LDA")

############# LDA Plot 3
#
# the curve for p(x) for the test data
# p(x)=sum(k=1,K)[(p(x|C=k)*p(C=k))] = sum(k=1,K)[p(x|C=k)*1/3]
#
probs2 = np.array(((probs)/3.0),dtype=np.float32)
px = np.sum(probs2,axis=1)[:,np.newaxis]
makePlot3(Xtests, px, "LDA")

############# LDA Plot 4
#
# the three curves for p(C=k|x) for k = 1, 2, and 3, for the test data
# p(C=k|x) = (p(x|C=k) * p(C=k))/p(x) = (probs1 * 1/3 ) / probs2 
#
probs3 = np.array(((probs/3.0)/px),dtype=np.float32)
makePlot4(Xtests, probs3, "LDA")

############# LDA Plot 5
#
# the three discriminant functions for the test data
#
plt.subplot(6,1,5)
plt.plot(Xtests[:,0],np.vstack((d1t,d2t)).T)
plt.ylabel("LDA"+" $\delta_{k}$")

############# LDA Plot 6
#
# the class predicted by the classifier for the test data
#
predictedTest = np.argmax(np.vstack((ld1t,ld2t)),axis=0)[:,np.newaxis]
# increment returned indices
predictedTest = predictedTest+1
plt.subplot(6,1,6)
plt.plot(Xtests[:,0], predictedTest, '_')
plt.ylim([-0.5,2.5])
plt.ylabel("LDA"+" predicted class")
plt.show()


############# LR Plot 1
#
# the training data, as its x value versus the class (1, 2 or 3)
#
plt.figure(1)
plt.clf()
plt.subplot(6,1,1)
Ttr = Ttrain.ravel()==0
X1=Xtrains[Ttr,:]
T1=Ttrain[Ttr,:]
Ttr = Ttrain.ravel()==1
X2=Xtrains[Ttr,:]
T2=Ttrain[Ttr,:]

plt.plot(X1, T1, 'bo')
plt.plot(X2, T2, 'ro')
plt.ylim([-0.5,1.5])
plt.ylabel("Train Class")

############# LR Plot 2
#
# the three curves for p(x|Class=k) for k=1,2,3, for x values in a set of test 
# data generated by x = np.linspace(0,4,100), where p(x|C=k) is calculated 
# using means and standard deviations for each class calculated from the 
# training data
#
prob1 = normald(standardize(Xtests), mu1, cov1)
prob2 = normald(standardize(Xtests), mu2, cov2)
probs = np.hstack((prob1,prob2))

plt.subplot(4,1,2)
plt.plot(Xtests[:,0],prob1)
plt.plot(Xtests[:,0],prob2)
plt.ylabel("LR"+" $p(x|class=k)$")  

############# LR Plot 3
#
# the class predicted by the classifier for the test data
#
plt.subplot(4,1,3)
plt.plot(Xtests[:,0], predictedTest, '_')
plt.ylim([-0.5,2.5])
plt.ylabel("LR"+" predicted class")


############# LR Plot 4
#
# plot the g's from LR
#
plt.subplot(4,1,4)
plt.plot(Xtests[:,0],logregOutput)
plt.ylabel("LR"+" $g(x)$")  
plt.show()

