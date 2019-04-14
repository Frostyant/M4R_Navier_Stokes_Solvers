from firedrake import *
from matplotlib import *
import numpy as np
import sys
sys.path.append('../')
from rins import rins

# Load mesh
mesh = Mesh("Cyl05.msh")

#defining
x,y= SpatialCoordinate(mesh)
#defining time
ts = np.arange(0.0,2.0,0.1)
t = Constant(ts[0])

# Define function spaces
V = FunctionSpace(mesh, "BDM", 2)
Q = FunctionSpace(mesh, "DG", 1)
W = V * Q
AverageVelocity = 1

# boundary function, these are assumed to not change during iteration
u_0 = as_vector([conditional(x < 5,AverageVelocity,0.)
    ,0])

problem = rins.rinspt(mesh,u_0,W,x,y,t,viscosity = 20,BcIds = (1,5), DbcIds = (1,3,4,5),AdvectionSwitchStep = 0.25,AverageVelocity = AverageVelocity,LengthScale = 2)

print("Reynolds Number =")
print(problem.R)

problem.StabTest(ts,order = -2, PicIt = 10)
