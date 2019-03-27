from matplotlib import *
import matplotlib.pyplot as plt
import numpy as np
import math

xs = np.array([0,1/2*math.pi,1*math.pi,3/2*math.pi,2*math.pi])
ys = np.sin(xs)

x2s = np.arange(0,2*math.pi,1/128*math.pi)
y2s = ns.sin(x2s)

plt.xlabel('x')
plt.ylabel('f')
plt.plot(xs,ys, label = "lg interpolation")
plt.plot(x2s,y2s, label = "sin(x)")
plt.title('1D Lagrange Interpolation')
plt.savefig('ex_1D_LG.png')
plt.figure()
