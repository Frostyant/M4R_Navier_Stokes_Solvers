from firedrake import *
from matplotlib import *
import numpy as np
import sys
sys.path.append('../')
from rins import rins


# Load mesh
mesh = Mesh("DoubleCircle.msh")

#defining
x,y= SpatialCoordinate(mesh)

# Define function spaces
V = FunctionSpace(mesh, "BDM", 2)
Q = FunctionSpace(mesh, "DG", 1)
W = V * Q
AverageVelocity = 1

# boundary function
u_0 = AverageVelocity*as_vector([
    conditional(x**2 + y**2 < 1.1**2,-y , 0.)
    + conditional(x**2 + y**2 > 1.1**2,-3*y , 0.) ,
    conditional(x**2 + y**2 < 1.9**2,x , 0.)
    + conditional(x**2 + y**2 > 1.9**2, 3*x , 0.)
    ])

problem = rins.rinsp(mesh,u_0,W,x,y,BcIds = (1,2),AverageVelocity = AverageVelocity,LengthScale = 4)

print("Reynolds Number =")
print(problem.R)

problem.FullSolve()
