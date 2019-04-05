from firedrake import *
from matplotlib import *
import numpy as np
import sys
sys.path.append('../')
from rins import rins
import matplotlib.pyplot as plt

AverageVelocity = 1
mu = 1

Ns = [2**(n+3) for n in range(7)]
errors = [0]*len(Ns)

for it,n in enumerate(Ns):
    mesh = UnitSquareMesh(n, n)
    x,y= SpatialCoordinate(mesh)
    V = FunctionSpace(mesh, "BDM", 2)
    Q = FunctionSpace(mesh, "DG", 1)
    W = V * Q
    u_0 = as_vector([mu + exp(y),mu + exp(x)])

    problem = rins.rinsp(mesh,u_0,W,x,y,viscosity = mu,BcIds = (1,2,3,4),AdvectionSwitchStep = 0.25,AverageVelocity = AverageVelocity,LengthScale = 1)
    problem.FullSolve(FullOutput = False,DisplayInfo = False,stokes = False)
    print("Reynolds Number =")
    print(problem.R)

    #dealing with stokes error
    u, p = problem.up.split()
    uexact = Function(V)
    uexact.project(u_0)
    errors[it] = norm(u-uexact)

plt.xlabel('h')
plt.ylabel('L1 Error')
plt.loglog(Ns,errors)
plt.title('Navier-Stokes Convergence Graph')
plt.savefig('navier_stokes_convergence.png')

#plotting error in space
ufile = File("error.pvd")
u, p = problem.up.split()
u -= uexact
u.rename("error")
ufile.write(u)
