
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
from ClassifierLib import *

N=20

means = [1.0,2.0,3.0]
sigma = 0.1
xs1 = np.random.normal(means[0],sigma,(N,1))
ys1 = np.random.normal(means[0],sigma,(N,1))
xs2 = np.random.normal(means[1],sigma,(N,1))
ys2 = np.random.normal(means[1],sigma,(N,1))
xs3 = np.random.normal(means[2],sigma,(N,1))
ys3 = np.random.normal(means[2],sigma,(N,1))
(x1,y1) = np.meshgrid(xs1, ys1)
(x2,y2) = np.meshgrid(xs2, ys2)
(x3,y3) = np.meshgrid(xs3, ys3)

Xtrain =  np.vstack((np.hstack((x1,x2,x3)), np.hstack((y1,y2,y3)))).T
standardize,unstandardize = makeStandardizeF(Xtrain)
Xtrains = standardize(Xtrain)
Ttrain = np.repeat(range(1,4),N)

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
xs = np.linspace(0.0,1.0,N)
x,y = np.meshgrid(xs,xs)
Xtest = np.vstack((x.flat,y.flat)).T
Xtests = standardize(Xtrain)
Ttest = Ttrain

############# QDA Plot 1
#
# the training data, as its x value versus the class (1, 2 or 3)
#
make2DPlot1(Xtrains, Ttrain)






colors = ['blue','red','green']
classes = [1.,2.,3.]

#prob1 = normald(Xtest, mu1, sigma1)
#prob2 = normald(Xtest, mu2, sigma2)
#prob3 = normald(Xtest, mu3, sigma3)
#probs=np.hstack((prob1,prob2,prob3))

############# p(C=k|x) plot
#fig = plt.figure(6)
#plt.clf()
#ax = Axes3D(fig)

#for k in range(K):
#    x = np.array(Xtests[:,0][:,np.newaxis])
#    y = np.array(Xtests[:,1][:,np.newaxis])
#    z = np.array(probs[:,k][:,np.newaxis]).reshape(x.shape)

#    print("x.shape "+str(x.shape))
#    print("y.shape "+str(y.shape))
#    print("z.shape "+str(z.shape))

#    ax.plot_surface(x,y,z,rstride=1,cstride=1,color=colors[k])

#ax.set_zlabel('$p(Class=k|x)$')
plt.show()

