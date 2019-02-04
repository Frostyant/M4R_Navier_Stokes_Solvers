from firedrake import *
from matplotlib import *
import numpy as np
import rins

# Load mesh
mesh = Mesh("CylinderInPipe.msh")

#defining
x,y= SpatialCoordinate(mesh)

# Define function spaces
V = FunctionSpace(mesh, "BDM", 2)
Q = FunctionSpace(mesh, "DG", 1)
W = V * Q
AverageVelocity = 1

# boundary function, these are assumed to not change during iteration
u_0 = as_vector([conditional(x < 1,AverageVelocity,0.)
    ,0])

problem = rins.rinsp(mesh,u_0,W,x,y,viscosity = 0.1,BcIds = (1,5),AverageVelocity = AverageVelocity,LengthScale = 50)

print("Reynolds Number =")
print(problem.R)

problem.FullSolve(FullOutput =True)
