from firedrake import *
from matplotlib import *
import numpy as np
import rins

# Load mesh
mesh = UnitSquareMesh(50, 50)

# Define function spaces
V = FunctionSpace(mesh, "BDM", 2)
Q = FunctionSpace(mesh, "DG", 1)
W = V * Q

#defining time
ts = np.arange(0.0,2.0*np.pi,0.5*np.pi)
t = Constant(ts[0])

#defining
x,y= SpatialCoordinate(mesh)

# boundary function, these are assumed to not change during iteration
u_0 = as_vector([conditional(y < 0.1,sin(t),0.),0])

problem = rins.rinspt(ts,mesh,u_0,W,x,y,t,BcIds = 3,V=V)

print("Reynolds Number =")
print(problem.R)

problem.SolveInTime()
