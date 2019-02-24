from firedrake import *
from matplotlib import *
import numpy as np
import sys
sys.path.append('../')
from rins import rins
import matplotlib.pyplot as plt

AverageVelocity = 1
mu = 1

Ns = [10**n for n in range(5)]
errors = [0]*len(Ns)

for it,n in enumerate(Ns):
    mesh = UnitSquareMesh(n, n)
    x,y= SpatialCoordinate(mesh)
    V = FunctionSpace(mesh, "BDM", 2)
    Q = FunctionSpace(mesh, "DG", 1)
    W = V * Q
    u_0 = as_vector([Constant(mu)+exp(y),Constant(mu)+exp(x)])

    problem = rins.rinsp(mesh,u_0,W,x,y,viscosity = mu,BcIds = (1,2,3,4),AdvectionSwitchStep = 0.25,AverageVelocity = AverageVelocity,LengthScale = 1)
    problem.FullSolve(FullOutput = False,DisplayInfo = False)
    print("Reynolds Number =")
    print(problem.R)

    #dealing with error
    u, p = problem.up.split()
    uexact = Function(V)
    uexact.project(u_0)
    errors[it] = norm(u-u_0)

plt.xlabel('nodes')
plt.ylabel('Hdiv error')
plt.plot(Ns,errors)
plt.title('Mesh Convergence Graph')
plt.savefig('Convergence.png')
