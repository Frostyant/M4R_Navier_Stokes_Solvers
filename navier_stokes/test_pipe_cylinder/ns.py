from firedrake import *
from matplotlib import *
import numpy as np
import sys
sys.path.append('../')
from rins import rins

# Load mesh
mesh = Mesh("Cyl1.msh")

#defining
x,y= SpatialCoordinate(mesh)

# Define function spaces
V = FunctionSpace(mesh, "BDM", 2)
Q = FunctionSpace(mesh, "DG", 1)
W = V * Q
AverageVelocity = 1

# boundary function, these are assumed to not change during iteration
u_0 = as_vector([conditional(x < 5,AverageVelocity,0.)
    ,0])

problem = rins.rinsp(mesh,u_0,W,x,y,viscosity = 0.024,BcIds = (1,5),AdvectionSwitchStep = 0.25,AverageVelocity = AverageVelocity,LengthScale = 1)

print("Reynolds Number =")
print(problem.R)

problem.FullSolve(FullOutput =True)
