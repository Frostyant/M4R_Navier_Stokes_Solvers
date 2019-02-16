from firedrake import *
from matplotlib import *
import numpy as np
import rins


# Load mesh
mesh = Mesh("DoubleCircle.msh")

#defining
x,y= SpatialCoordinate(mesh)

# Define function spaces
V = FunctionSpace(mesh, "BDM", 2)
Q = FunctionSpace(mesh, "DG", 1)
W = V * Q
AverageVelocity = 1

#defining time
ts = np.arange(0.0,1.0,0.01)
t = Constant(ts[0])

# boundary function, these are assumed to not change during iteration
u_0 = as_vector([conditional(x**2 + y**2 < 1.1,AverageVelocity,0.)
    ,0])


problem = rins.rinspt(mesh,u_0,W,x,y,t,BcIds = (1),AverageVelocity = AverageVelocity,LengthScale = 4)

print("Reynolds Number =")
print(problem.R)

problem.SolveInTime(ts)
