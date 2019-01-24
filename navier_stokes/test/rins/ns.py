from firedrake import *
from matplotlib import *
import numpy as np
import rins

# Load mesh
mesh = Mesh("CylinderInPipe.msh")

# Define function spaces
V = FunctionSpace(mesh, "BDM", 2)
Q = FunctionSpace(mesh, "DG", 1)
W = V * Q
#defining
x,y= SpatialCoordinate(mesh)
#defining the normal
n = FacetNormal(mesh)

# boundary function, these are assumed to not change during iteration
u_0 = as_vector([conditional(x < 1,1,0.)
    ,0])

#Quick Explanation: We cannot solve for high reynolds number, so instead we solve for low viscosity and then gradually increaseself.
#We then use the u estimated from the prior step as a guess for the next one so that the solver converges.
#In theory firedrake should automatically use the prior u values as a guess since they are stored in the variable which generates the new test fct.

#Bc1
bc1 = DirichletBC(W.sub(0), u_0, 1) #Can only set Normal Component, here that is u left bdary
#bc1p = DirichletBC(W.sub(1), p_0, 1)

#Bc2
#bc2 = DirichletBC(W.sub(0), u_0, 2)#Can only set Normal Component, here that is u right bdary

#Bc3
bc3 = DirichletBC(W.sub(0), u_0, 3)#Can only set Normal Component, here that is v bottom bdary

#Bc4
bc4 = DirichletBC(W.sub(0), u_0, 4)#Can only set Normal Component, here that is v top bdary

#Bc5, From cylinder
bc5 = DirichletBC(W.sub(0), u_0, 5)

#boundary conditions
bcs=(bc1,bc3,bc4,bc5)

problem = rins.rinsp(mesh,u_0,bcs,W,x,y)

problem.FullSolve()
