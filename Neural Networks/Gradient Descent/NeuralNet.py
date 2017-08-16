import gradientDescent as gd
reload(gd)
import numpy as np
import matplotlib.pylab as plt
from math import sqrt
from copy import deepcopy

class NeuralNet:
    def __init__(self,X,T,nh=10,nnet=None,
                 nIterations=100,weightPrecision=0.0001,errorPrecision=0.0001):
        if not nnet:
            self.standardize = makeStandardizeF(X)[0]
            (self.standardizeT,self.unstandardizeT) = makeStandardizeF(T)
            self.ni = X.shape[1]
            self.nh = nh
            self.no = T.shape[1]
            # Initialize weights to uniformly distributed values between small normally-distributed between -0.1 and 0.1
            self.V = np.random.uniform(-0.1,0.1,size=(1+self.ni,self.nh))
            self.W = np.random.uniform(-0.1,0.1,size=(1+self.nh,self.no))
        else:
            self = deepcopy(nnet)
    
        self.X = self.standardize(X)
        self.T = self.standardizeT(T)
        self.X1 = addOnes(self.X)
        
        weights = self.pack(self.V, self.W)
        scgresult = gd.scg(weights, self.mseTrainF, self.gradF, self.X1, self.T, xPrecision=weightPrecision, fPrecision=errorPrecision, xtracep=True, ftracep=True)

        self.unpack(scgresult['x'])
        self.reason = scgresult['reason']
        self.errorTrace = scgresult['ftrace']

    def pack(self,dV,dW):
        return np.hstack((dV.flat,dW.flat))

    def unpack(self,w):
        self.V = w[0:((self.ni+1)*self.nh)]
        self.V.resize((self.ni+1,self.nh))
        self.W = w[((self.ni+1)*self.nh):]
        self.W.resize((self.nh+1,self.no))

    def mseTrainF(self,w,X1,T):
        self.unpack(w)
        Z = np.tanh(np.dot( X1, self.V ))
        Z1 = addOnes(Z)
        Y = np.dot( Z1, self.W )
        return 0.5 * np.mean(((Y - T)**2).flat)

    def gradF(self,w,X1,T):
        self.unpack(w)
        Z = np.tanh(np.dot(X1, self.V))
        Z1 = addOnes(Z)
        Y = np.dot(Z1, self.W)
        error = (Y - T) / (X1.shape[0] * T.shape[1])
        dV = np.dot( X1.T, np.dot( error, self.W[1:,:].T) * (1-Z**2))
        dW = np.dot( Z1.T, error)
        return self.pack(dV,dW)

    def use(self,X,allOutputs=False):
        X1 = addOnes(X)
        Z = np.tanh(np.dot(X1, self.V))
        Z1 = addOnes(Z)
        Y = np.dot(Z1, self.W)
        Y = self.unstandardizeT(Y)
        return (Z,Y) if allOutputs else Y

    def plotError(self):
        plt.plot(self.errorTrace)
        plt.xlabel("Iteration")
        plt.ylabel("Train MSE")

    def draw(self, inputNames = None, outputNames = None, gray = False):
    
        def isOdd(x):
            return x % 2 != 0

        W = [self.V, self.W]
        nLayers = 2

        # calculate xlim and ylim for whole network plot
        #  Assume 4 characters fit between each wire
        #  -0.5 is to leave 0.5 spacing before first wire
        xlim = max(map(len,inputNames))/4.0 if inputNames else 1
        ylim = 0
    
        for li in range(nLayers):
            ni,no = W[li].shape  #no means number outputs this layer
            if not isOdd(li):
                ylim += ni + 0.5
            else:
                xlim += ni + 0.5

        ni,no = W[nLayers-1].shape  #no means number outputs this layer
        if isOdd(nLayers):
            xlim += no + 0.5
        else:
            ylim += no + 0.5

        # Add space for output names
        if outputNames:
            if isOdd(nLayers):
                ylim += 0.25
            else:
                xlim += round(max(map(len,outputNames))/4.0)

        ax = plt.gca()

        x0 = 1
        y0 = 0 # to allow for constant input to first layer
        # First Layer
        if inputNames:
            #addx = max(map(len,inputNames))*0.1
            y = 0.55
            for n in inputNames:
                y += 1
                ax.text(x0-len(n)*0.2, y, n)
                x0 = max([1,max(map(len,inputNames))/4.0])

        for li in range(nLayers):
            Wi = W[li]
            ni,no = Wi.shape
            if not isOdd(li):
                # Odd layer index. Vertical layer. Origin is upper left.
                # Constant input
                ax.text(x0-0.2, y0+0.5, '1')
                for li in range(ni):
                    ax.plot((x0,x0+no-0.5), (y0+li+0.5, y0+li+0.5),color='gray')
                # output lines
                for li in range(no):
                    ax.plot((x0+1+li-0.5, x0+1+li-0.5), (y0, y0+ni+1),color='gray')
                # cell "bodies"
                xs = x0 + np.arange(no) + 0.5
                ys = np.array([y0+ni+0.5]*no)
                ax.scatter(xs,ys,marker='v',s=1000,c='gray')
                # weights
                if gray:
                    colors = np.array(["black","gray"])[(Wi.flat >= 0)+0]
                else:
                    colors = np.array(["red","green"])[(Wi.flat >= 0)+0]
                xs = np.arange(no)+ x0+0.5
                ys = np.arange(ni)+ y0 + 0.5
                aWi = abs(Wi)
                aWi = aWi / np.max(aWi) * 50
                coords = np.meshgrid(xs,ys)
                #ax.scatter(coords[0],coords[1],marker='o',s=2*np.pi*aWi**2,c=colors)
                ax.scatter(coords[0],coords[1],marker='s',s=aWi**2,c=colors)
                y0 += ni + 1
                x0 += -1 ## shift for next layer's constant input
            else:
                # Even layer index. Horizontal layer. Origin is upper left.
                # Constant input
                ax.text(x0+0.5, y0-0.2, '1')
                # input lines
                for li in range(ni):
                    ax.plot((x0+li+0.5,  x0+li+0.5), (y0,y0+no-0.5),color='gray')
                # output lines
                for li in range(no):
                    ax.plot((x0, x0+ni+1), (y0+li+0.5, y0+li+0.5),color='gray')
                # cell "bodies"
                xs = np.array([x0 + ni + 0.5]*no)
                ys = y0 + 0.5 + np.arange(no)
                ax.scatter(xs,ys,marker='>',s=1000,c='gray')
                # weights
                Wiflat = Wi.T.flatten()
                print [w for w in Wi.flat]
                if gray:
                    colors = np.array(["black","gray"])[(Wiflat >= 0)+0]
                else:
                    colors = np.array(["red","green"])[(Wiflat >= 0)+0]
                xs = np.arange(ni)+x0 + 0.5
                ys = np.arange(no)+y0 + 0.5
                coords = np.meshgrid(xs,ys)
                aWi = abs(Wiflat)
                aWi = aWi / np.max(aWi) * 50
                #ax.scatter(coords[0],coords[1],marker='o',s=2*np.pi*aWi**2,c=colors)
                ax.scatter(coords[0],coords[1],marker='s',s=aWi**2,c=colors)
                x0 += ni + 1
                y0 -= 1 ##shift to allow for next layer's constant input

        # Last layer output labels 
        if outputNames:
            if isOdd(nLayers):
                x = x0+1.5
                for n in outputNames:
                    x += 1
                    ax.text(x, y0+0.5, n)
            else:
                y = y0+0.6
                for n in outputNames:
                    y += 1
                    ax.text(x0+0.2, y, n)
        ax.axis([0,xlim, ylim,0])
        ax.axis('off')

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

if __name__== "__main__":
    X = np.array([[0,0],[1,0],[0,1],[1,1]])
    T = np.array([[0],[1],[1],[0]])
    nnet = NeuralNet(X,T,10,weightPrecision=0,errorPrecision=0,nIterations=1000)
    print "SCG stopped after",len(nnet.errorTrace),"iterations:",nnet.reason
    y = nnet.use(X)
    
    print "Inputs"
    print X
    print "Targets"
    print T
    
    print np.hstack((T,y))

    #plt.ion()
    plt.figure(1)
    plt.clf()
    plt.subplot(2,1,1)
    nnet.plotError()
    plt.subplot(2,1,2)
    nnet.draw(['x1','x2'],['xor'])
    plt.show()
