from firedrake import *
from matplotlib import *
import numpy as np
import sys
sys.path.append('../')
from rins import rins
import matplotlib.pyplot as plt

AverageVelocity = 1
mu = 1

Ns = [2**(n+4) for n in range(6)]
errors = [0]*len(Ns)

for it,n in enumerate(Ns):
    mesh = UnitSquareMesh(n, n)
    x,y= SpatialCoordinate(mesh)
    V = FunctionSpace(mesh, "BDM", 2)
    Q = FunctionSpace(mesh, "DG", 1)
    W = V * Q
    u_0 = as_vector([sin(2*pi*x)*sin(2*pi*y),(1-cos(2*pi*x))*sin(2*pi*y)])
    u_x = u_0.dx(0)
    u_y = u_0.dx(1)
    F = -mu*as_vector([ (u_x.dx(0)[0] + u_y.dx(1)[0]) , (u_x.dx(0)[1] + u_y.dx(1)[1]) ])

    problem = rins.rinsp(mesh,u_0,W,x,y,viscosity = mu,BcIds = (1,2,3,4),AdvectionSwitchStep = 0.25,AverageVelocity = AverageVelocity,LengthScale = 1)
    problem.FullSolve(FullOutput = False,DisplayInfo = False,stokes = True)
    print("Reynolds Number =")
    print(problem.R)

    #dealing with stokes error
    u, p = problem.up.split()
    uexact = Function(V)
    uexact.project(u_0)
    errors[it] = norm(u-uexact)

plt.xlabel('o(n)')
plt.ylabel('L1 Error')
plt.loglog(Ns,errors)
plt.title('Stokes Convergence Graph')
plt.savefig('stokes_convergence.png')

#plotting error in space
ufile = File("error.pvd")
u, p = problem.up.split()
u -= uexact
u.rename("error")
ufile.write(u)

valfile = open("stokes_error.txt","w+")
errorstring = ';'.join(str(e) for e in errors)
valfile.write(errorstring)
valfile.close()
