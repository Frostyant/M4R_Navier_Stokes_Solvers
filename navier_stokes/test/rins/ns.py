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
ts = np.arange(0,10,1)
t = Constant(ts[0])

#defining
x,y= SpatialCoordinate(mesh)

# boundary function, these are assumed to not change during iteration
u_0 = as_vector([conditional(y < 0.1,sin(t),0.),0])

#Bc1
#bc1 = DirichletBC(W.sub(0), u_0, 1) #Can only set Normal Component, here that is u left bdary

#Bc2
#bc2 = DirichletBC(W.sub(0), u_0, 2)#Can only set Normal Component, here that is u right bdary

#Bc3
bc3 = DirichletBC(W.sub(0), u_0, 3)#Can only set Normal Component, here that is v bottom bdary

#Bc4
#bc4 = DirichletBC(W.sub(0), u_0, 4)#Can only set Normal Component, here that is v top bdary

#boundary conditions
bcs=(bc3)

problem = rins.rinspt(ts,mesh,u_0,bcs,W,x,y,t,BcIds = (1))

problem.SolveInTime()
