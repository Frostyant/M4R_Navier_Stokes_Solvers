from firedrake import *
from matplotlib import *
import numpy as np
import rins

# Load mesh
mesh = UnitSquareMesh(50, 50)

# Load mesh
mesh = Mesh("CylinderInPipe.msh")

#defining
x,y= SpatialCoordinate(mesh)

# Define function spaces
V = FunctionSpace(mesh, "BDM", 2)
Q = FunctionSpace(mesh, "DG", 1)
W = V * Q
AverageVelocity = 1

#defining time
ts = np.arange(0.0,2.0,0.05)
t = Constant(ts[0])

# boundary function, these are assumed to not change during iteration
u_0 = as_vector([conditional(x < 3,AverageVelocity,0.)
    ,0])


problem = rins.rinspt(mesh,u_0,W,x,y,t,BcIds = 3,AverageVelocity = AverageVelocity,LengthScale = 50)

print("Reynolds Number =")
print(problem.R)

problem.SolveInTime(ts)
