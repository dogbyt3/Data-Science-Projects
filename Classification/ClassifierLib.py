import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def discQDA(X, standardize, mu, Sigma, prior):
    Xc = standardize(X) - mu
    if Sigma.size == 1:
        Sigma = np.asarray(Sigma).reshape((1,1))
    SigmaInv = np.linalg.inv(Sigma)
    return -0.5 * np.log(np.linalg.det(Sigma)) \
           - 0.5 * np.sum(np.dot(Xc,SigmaInv) * Xc, axis=1) \
           + np.log(prior)

def discLDA(X, standardize, mu, Sigma, prior):
    Xc = standardize(X)
    if Sigma.size == 1:
        Sigma = np.asarray(Sigma).reshape((1,1))
    SigmaInv = np.linalg.inv(Sigma)
    return np.dot(np.dot(Xc,SigmaInv), mu.T)\
           -0.5 * np.dot(np.dot(mu.T, SigmaInv), mu)\
           + np.log(prior)

def g(X,beta):
    fs = np.exp(np.dot(X, beta))  # N x K-1
    denom = 1 + np.sum(fs,axis=1).reshape(-1,1) #reshape to preserve 2d shape 
    gs = fs / denom
    return np.hstack((gs,1/denom))

def makeIndicatorVars(X):
    """X is a column vector"""
    X = np.asanyarray(X)
    X.resize((X.size,1))
    return 0 + (X == np.unique(X))

def percentCorrect(p,t):
    return np.sum(p.ravel()==t.ravel()) / float(len(t)) * 100

# For multiple samples, for any dimension, including 1
def normald(X, mu=None, sigma=None):
    """ normald:
       X contains samples, one per row, NxD. 
       mu is mean vector, Dx1.
       sigma is covariance matrix, DxD.  """
    d = X.shape[1]
    if np.any(mu == None):
        mu = np.zeros((d,1))
    if np.any(sigma == None):
        sigma = np.eye(d)
    if d == 1:
        detSigma = sigma
        sigmaI = 1.0/sigma
    else:
        detSigma = np.linalg.det(sigma)
        sigmaI = np.linalg.inv(sigma)
    normConstant = 1.0 / np.sqrt((2*np.pi)**d * detSigma)
    diffv = X - mu.T # change column vector mu to be row vector
    return normConstant * \
           np.exp(-0.5 * np.sum(np.dot(diffv, sigmaI) * diffv, axis=1)) \
           [:,np.newaxis]

def pClassGivenX(X, mus, sigmas, standardize):
    return normald(standardize(X), mu, sigma)

def addOnes(X):
    return np.hstack((np.ones((X.shape[0],1)), X))

def makeStandardizeF(X):
    means = X.mean(axis=0)
    stds = X.std(axis=0,ddof=1)
    def standardize(origX):
        return (origX - means) / stds
    def unStandardize(stdX):
        return stds * stdX + means
    return (standardize, unStandardize)

def mvNormalRand(n,mean,sigma):                                                 
    mean = np.array(mean)                                                       
    sigma = np.array(sigma)                                                     
    X = np.random.normal(0,1,n*len(mean)).reshape((n,len(mean)))                
    return np.dot(X, np.linalg.cholesky(sigma)) + mean

############# Plot 1
#
# the training data, as its x value versus the class (1, 2 or 3)
#
def makePlot1(X1, X2, X3, T1, T2, T3):
    plt.plot(X1, T1, 'bo')
    plt.plot(X2, T2, 'ro')
    plt.plot(X3, T3, 'go')
    plt.ylim([0.5,3.5])
    plt.xlim([0,4])
    plt.ylabel("Train Class")

############# Plot 2
#
# the three curves for p(x|Class=k) for k=1,2,3, for x values in a set of test 
# data generated by x = np.linspace(0,4,100), where p(x|C=k) is calculated 
# using means and standard deviations for each class calculated from the 
# training data
#
def makePlot2(newData, probs, disc):
    plt.subplot(6,1,2)
    plt.plot(newData[:,0],probs)
    plt.ylabel(disc+" $p(x|class=k)$")  

############# Plot 2
#
# the three curves for p(x|Class=k) for k=1,2,3, for x values in a set of test 
# data generated by x = np.linspace(0,4,100), where p(x|C=k) is calculated 
# using means and standard deviations for each class calculated from the 
# training data
#
#def makePlot2(newData, prob1, prob2, prob3, disc):
#    plt.subplot(6,1,2)
#    plt.plot(newData[:,0],prob1)
#    plt.plot(newData[:,0],prob2)
#    plt.plot(newData[:,0],prob3)
#    plt.ylabel(disc+" $p(x|class=k)$")  

############# Plot 3
#
# the curve for p(x) for the test data
# p(x)=sum(k=1,K)[(p(x|C=k)*p(C=k))] = p(x|C=k)*1/3
#    
def makePlot3(newData, probs2, disc):
    plt.subplot(6,1,3)
    plt.plot(newData[:,0],probs2)
    plt.ylabel(disc+" $p(x)$")

############# Plot 4
#
# the three curves for p(C=k|x) for k = 1, 2, and 3, for the test data
# p(C=k|x) = (p(x|C=k) * p(C=k))/p(x) = (probs1 * 1/3 ) / probs2 
#
def makePlot4(newData, probs3, disc):
    plt.subplot(6,1,4)
    plt.plot(newData[:,0],probs3)        
    plt.ylim([-0.5,1.5])
    plt.ylabel(disc+" $p(class=k|x)$")

############# Plot 5
#
# the three discriminant functions for the test data
#
def makePlot5(newData, d1, d2, d3, disc):
    plt.subplot(6,1,5)
    plt.plot(newData[:,0],np.vstack((d1,d2,d3)).T)
    plt.ylabel(disc+" $\delta_{k}$")

############# Plot 6
#
# the class predicted by the classifier for the test data
#
def makePlot6(newData, d1, d2, d3, disc):
    predictedTest = np.argmax(np.vstack((d1,d2,d3)),axis=0)[:,np.newaxis]
    # increment returned indices
    predictedTest = predictedTest+1
    plt.subplot(6,1,6)
    plt.plot(newData[:,0], predictedTest, 'k_')
    plt.ylabel(disc+" predicted class")
    plt.ylim([-0.5,3.5])

############# LR Plot 2
#
# the three curves for p(x|Class=k) for k=1,2,3, for x values in a set of test 
# data generated by x = np.linspace(0,4,100), where p(x|C=k) is calculated 
# using means and standard deviations for each class calculated from the 
# training data
#
def makeLRPlot2(newData, prob1, prob2, prob3, disc):
    plt.subplot(4,1,2)
    plt.plot(newData[:,0],prob1)
    plt.plot(newData[:,0],prob2)
    plt.plot(newData[:,0],prob3)
    plt.ylabel(disc+" $p(x|class=k)$")  

############# LR Plot 3
#
# the class predicted by the classifier for the test data
#    
def makeLRPlot3(newData, predictedTest, disc):
    plt.subplot(4,1,3)
    plt.plot(newData[:,0], predictedTest, 'k_')
    plt.ylabel(disc+" predicted class")
    plt.ylim([-0.5,3.5])

############# LR Plot 4
#
# plot the g's from LR
#
def makeLRPlot4(newData, gs, disc):
    plt.subplot(4,1,4)
    plt.plot(newData[:,0],gs)
    plt.ylabel(disc+" $g(x)$")  


############# 2D Plot 1
#
# the training data, as its x value versus the class (1, 2 or 3)
#
def make2DPlot1(Xtrain, Ttrain, classes=[1,2,3], K=3):
    fig = plt.figure(3)
    plt.clf()
    ax = Axes3D(fig)
    colors = ['blue','red','green']
    for k in range(K):
        r = (Ttrain == classes[k]).flatten()
        Xk = Xtrain[r,:]
        ax.plot3D(Xk[:,0].ravel(),Xk[:,1].ravel(), Ttrain[r,:].ravel(), 'o',color=colors[k])
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_zlabel('Train Class')

############# 2D Plot 2
#
# the training data versus the class (1, 2 or 3)
#
def make2DPlot2(newData, Xtrain, Ttrain, prob, K=3):
    fig = plt.figure(6)
    plt.clf()
    ax = Axes3D(fig)
    for k in range(K):
        ax.plot_surface(x,y, pClassGivenX[:,k].reshape(x.shape),\
                        rstride=1,cstride=1,color=colors[k])
    ax.set_zlabel('$p(Class=k|x)$')

def pClassGivenX(X, mus, sigmas, standardize):
    prob1 = normald(standardize(X1), mu1, sigma1)
    prob2 = normald(standardize(X2), mu2, sigma2)
    prob3 = normald(standardize(X3), mu3, sigma3)
    probs=np.hstack((prob1,prob2,prob3))

    return normald(standardize(X), mu, sigma)

#    crash
    

#    ax.plot_surface(x,y, prob1, rstride=1,cstride=1,color='blue')
#    ax.plot_surface(newData, prob2, rstride=1,cstride=1,color='red')
#    ax.plot_surface(newData, prob3, rstride=1,cstride=1,color='green')
#    ax.set_zlabel('$p(Class=k|x)$')
