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
ep = [0]*len(Ns)

for it,n in enumerate(Ns):
    mesh = UnitSquareMesh(n, n)
    x,y= SpatialCoordinate(mesh)
    V = FunctionSpace(mesh, "BDM", 2)
    Q = FunctionSpace(mesh, "DG", 1)
    W = V * Q
    #u_0 = as_vector([-x*sin(2*pi*x*y),y*sin(2*pi*x*y)])
    #p_0 = sin(x*y)
    u_0 = as_vector([-sin(2*pi*y)*cos(2*pi*y)*sin(2*pi*x)**2,sin(2*pi*x)*cos(2*pi*x)*sin(2*pi*y)**2])
    u_1 = as_vector([-sin(2*pi*y)*cos(2*pi*y)*sin(2*pi*x)**2,sin(2*pi*x)*cos(2*pi*x)*sin(2*pi*y)**2])
    p_0 = Constant(1)*sin(2*pi*x)**2*sin(2*pi*y)**2
    p_x = p_0.dx(0)
    p_y = p_0.dx(1)
    u_xx = u_0.dx(0).dx(0)
    u_yy = u_0.dx(1).dx(1)
    F_ = as_vector([p_x -mu*(u_xx[0] + u_yy[0]) , p_y -mu*(u_xx[1] + u_yy[1]) ])
    u_x = u_1.dx(0)
    u_y = u_1.dx(1)
    F_adv = as_vector([ u_0[0]*u_x[0]+u_0[1]*u_y[0], u_0[0]*u_x[1]+u_0[1]*u_y[1] ])
    F = Function(V)
    F.project(F_)# + F_adv)

    problem = rins.rinsp(mesh,u_0,W,x,y, F = F,viscosity = mu,BcIds = (1,2,3,4),AdvectionSwitchStep = 0.25,AverageVelocity = AverageVelocity,LengthScale = 1)
    problem.FullSolve(FullOutput = False,DisplayInfo = False,stokes = True)#,method = "direct")
    print("Reynolds Number =")
    print(problem.R)

    #dealing with stokes error
    u, p = problem.up.split()
    uexact = Function(V)
    pexact = Function(Q)
    uexact.project(u_0)
    pexact.project(p_0)
    errors[it] = norm(u-uexact)
    ep[it] = norm(p-pexact)-0.25 #adjusting constant

plt.xlabel('o(n)')
plt.ylabel('L1 Error')
plt.loglog(Ns,errors)
plt.title('Stokes Velocity Convergence Graph')
plt.savefig('stokes_convergence.png')

plt.figure()
plt.xlabel('o(n)')
plt.ylabel('L1 Error')
plt.loglog(Ns,ep)
plt.title('Stokes Pressure Convergence Graph')
plt.savefig('stokes_pressure_convergence.png')

#plotting error in space
ufile = File("error.pvd")
u, p = problem.up.split()
u -= uexact
p -= pexact
u.rename("velocity error")
p.rename("pressure error")
ufile.write(u,p)

#plotting true solution in space
truefile = File("true.pvd")
uexact.rename("true velocity")
pexact.rename("true pressure")
truefile.write(uexact,pexact)

#saving exact values of the error
valfile = open("stokes_error.txt","w+")
errorstring = ';'.join(str(e) for e in errors)
valfile.write(errorstring)
valfile.close()

#saving exact values of the error
valfile2 = open("stokes_pressure_error.txt","w+")
errorstring2 = ';'.join(str(e) for e in ep)
valfile2.write(errorstring2)
valfile2.close()
