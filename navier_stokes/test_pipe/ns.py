from firedrake import *
from matplotlib import *
import numpy as np

#Some settings
n = 50
AverageVelocity = Constant(1)
viscosity = Constant(0.01)

mesh = UnitSquareMesh(n, n)
V = FunctionSpace(mesh, "BDM", 2)
Q = FunctionSpace(mesh, "DG", 1)
W = V * Q
x,y= SpatialCoordinate(mesh)
n = FacetNormal(mesh)

# boundary function, these are assumed to not change during iteration
u_0 = as_vector([conditional(x <0.1,AverageVelocity*sin(pi*y)**2,0.) + conditional(x > 0.9,AverageVelocity*sin(pi*y)**2,0.),0])
p_0 = 0

# boundary for penalty
u_0 = as_vector([10/(4*viscosity)*(0.5**2-(0.5-y)**2),0])#Pouseilles flow solution
p_0 = 0
p_1 = 10
#Bc1
bc1 = DirichletBC(W.sub(0), u_0, 1)
bc1p = DirichletBC(W.sub(1), p_0, 1)

#Bc2
bc2 = DirichletBC(W.sub(0), u_0, 2)

#Bc3
bc3 = DirichletBC(W.sub(1), p_1, 3)
#Bc4
bc4 = DirichletBC(W.sub(0), u_0, 4)

problem = rins.rinsp(mesh,u_0,W,x,y,viscosity = viscosity,BcIds = (1,2,4),AdvectionSwitchStep = 0.25,AverageVelocity = AverageVelocity,LengthScale = 1)

problem.FullSolve(FullOutput = False,DisplayInfo = False,stokes = False)
