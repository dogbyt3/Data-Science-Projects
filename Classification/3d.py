from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt



xs = np.linspace(0,10,num=20)
ys = np.linspace(0,10,num=20)
x,y = np.meshgrid(xs,ys)
z = np.linspace(0.0,1.0,num=20)
z,_ =  np.meshgrid(z, z)

print("x.shape "+str(x.shape))
print("y.shape "+str(y.shape))
print("z.shape "+str(z.shape))



fig = plt.figure(6)
plt.clf()
ax = Axes3D(fig)
ax.plot_surface(x,y,z,rstride=1,cstride=1,color='red')
ax.set_zlabel('$p(Class=k|x)$')
plt.show()
