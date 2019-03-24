from matplotlib import *
import matplotlib.pyplot as plt
import numpy as np
import math

xs = np.array([0,1/2*math.pi,1*math.pi,3/2*math.pi,2*math.pi])
ys = np.sin(xs)

plt.xlabel('x')
plt.ylabel('f')
plt.plot(xs,ys)
plt.title('1D Lagrange Interpolation')
plt.savefig('ex_1D_LG.png')
