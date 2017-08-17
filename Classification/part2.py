from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
from ClassifierLib import *

D = 2  # number of dimensions
N = 10 # sample number
K = 3  # number of classes

means = [1.0,2.0,3.0]
sigma = 0.1
xs1 = np.random.normal(means[0],sigma,(N,D-1))
ys1 = np.random.normal(means[0],sigma,(N,D-1))
xs2 = np.random.normal(means[1],sigma,(N,D-1))
ys2 = np.random.normal(means[1],sigma,(N,D-1))
xs3 = np.random.normal(means[2],sigma,(N,D-1))
ys3 = np.random.normal(means[2],sigma,(N,D-1))
(x1,y1) = np.meshgrid(xs1, ys1)
(x2,y2) = np.meshgrid(xs2, ys2)
(x3,y3) = np.meshgrid(xs3, ys3)

Xtrain =  np.vstack((np.hstack((x1,x2,x3)), np.hstack((y1,y2,y3)))).T
Ttrain = np.repeat(range(1,4),N)

standardize,unstandardize = makeStandardizeF(Xtrain)
Xtrains = standardize(Xtrain)

class1rows = Ttrain==1
class2rows = Ttrain==2
class3rows = Ttrain==3

mu1 = np.mean(Xtrains[class1rows,:],axis=0)
mu2 = np.mean(Xtrains[class2rows,:],axis=0)
mu3 = np.mean(Xtrains[class3rows,:],axis=0)

sigma1 = np.cov(Xtrains[class1rows,:].T)
sigma2 = np.cov(Xtrains[class2rows,:].T)
sigma3 = np.cov(Xtrains[class3rows,:].T)

N1 = np.sum(class1rows)
N2 = np.sum(class2rows)
N3 = np.sum(class3rows)

prior1 = N1/float(N)
prior2 = N2/float(N)
prior3 = N3/float(N)

# generate some new data
nNew = 10
xs1 = np.random.normal(means[0],sigma,(nNew,D-1))
ys1 = np.random.normal(means[0],sigma,(nNew,D-1))
xs2 = np.random.normal(means[1],sigma,(nNew,D-1))
ys2 = np.random.normal(means[1],sigma,(nNew,D-1))
xs3 = np.random.normal(means[2],sigma,(nNew,D-1))
ys3 = np.random.normal(means[2],sigma,(nNew,D-1))
(x1,y1) = np.meshgrid(xs1, ys1)
(x2,y2) = np.meshgrid(xs2, ys2)
(x3,y3) = np.meshgrid(xs3, ys3)

Xtest =  np.vstack((np.hstack((x1,x2,x3)), np.hstack((y1,y2,y3)))).T
Ttest = np.repeat(range(1,4),nNew)

print("Xtest.shape"+str(Xtest.shape))
print("Xtrain.shape"+str(Xtrain.shape))


# apply QDA discr to dat
d1 = discQDA(Xtest,standardize,mu1,sigma1,prior1)
d2 = discQDA(Xtest,standardize,mu2,sigma2,prior2)
d3 = discQDA(Xtest,standardize,mu3,sigma3,prior3)




###############################################################################
########## QDA Plots
###############################################################################

############# QDA Plot 1
#
# the training data, as its x value versus the class (1, 2 or 3)
#
make2DPlot1(Xtrains, Ttrain)

############# QDA Plot 2
#
# the three curves for p(x|Class=k) for k=1,2,3, for x values in a set of test 
# data generated by x = np.linspace(0,4,100), where p(x|C=k) is calculated 
# using means and standard deviations for each class calculated from the 
# training data
#
print("Xtrains.shape: "+str(Xtrains.shape))
print("Xtests.shape: "+str(Xtests.shape))

fig = plt.figure(6)
plt.clf()
ax = Axes3D(fig)
for k in range(K):
    ax.plot_surface(Xtests[:,0],Xtests[:,1], pClassGivenX[:,k].reshape(x.shape),rstride=1,cstride=1,color=colors[k])
ax.set_zlabel('$p(Class=k|x)$')


#prob1 = normald(standardize(X1), mu1, sigma1)
#prob2 = normald(standardize(X2), mu2, sigma2)
#prob3 = normald(standardize(X3), mu3, sigma3)
#probs=np.hstack((prob1,prob2,prob3))

#probs = np.exp(np.vstack((d1t-np.log(prior1),d2t-np.log(prior2),\
#                          d3t-np.log(prior3))).T \
#                          - 0.5*D*np.log(2*np.pi))
#makeQPlot2(newData, probs, "QDA")

#print("X1.shape: "+str(X1.shape))
#print("X.shape: "+str(X.shape))
#print("T.shape: "+str(T.shape))
#print("prob1.shape: "+str(prob1.shape))
#print("probs.shape: "+str(probs.shape))


#make2DPlot2(X1, X, T, prob1)
plt.show()