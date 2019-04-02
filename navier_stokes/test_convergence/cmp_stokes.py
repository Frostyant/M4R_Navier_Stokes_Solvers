from firedrake import *
from matplotlib import *
import numpy as np
import sys
sys.path.append('../')
from rins import rins
import matplotlib.pyplot as plt
import time

AverageVelocity = 1
mu = 1
n = 64

mesh = UnitSquareMesh(n, n)
x,y= SpatialCoordinate(mesh)
V = FunctionSpace(mesh, "BDM", 2)
Q = FunctionSpace(mesh, "DG", 1)
W = V * Q

u_0 = as_vector([cos(y)*exp(x),-sin(y)*exp(x)])

start = time.time()
problem = rins.rinsp(mesh,u_0,W,x,y,viscosity = mu,BcIds = (1,2,3,4),AdvectionSwitchStep = 0.25,AverageVelocity = AverageVelocity,LengthScale = 1)
problem.FullSolve(FullOutput = False,DisplayInfo = False,stokes = True)
end = time.time()
RinsT = end-start

#dealing with stokes error
u, p = problem.up.split()
uexact = Function(V)
uexact.project(u_0)
error1 = np.log(norm(u-uexact))








V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)
W = V*Q

#setting up boundary conditions
u_0 = as_vector([cos(y)*exp(x),-sin(y)*exp(x)])
start = time.time()
bc0 = DirichletBC(W.sub(0), u_0, 1)
bc1 = DirichletBC(W.sub(0), u_0, 2)
bc2 = DirichletBC(W.sub(0), u_0, 3)
bc3 = DirichletBC(W.sub(0), u_0, 4)
bcs = [bc0,bc1,bc2,bc3]

#Setting up functions
up = Function(W)
u, p = TrialFunctions(W)
(v, q) = TestFunctions(W)

# Removing Pressure constant
nullspace = MixedVectorSpaceBasis(
        W, [W.sub(0), VectorSpaceBasis(constant=True)])

#setting up problem proper
LHS = inner(grad(u), grad(v)) * dx + div(v) * p * dx + q * div(u) * dx
Zero = Function(Q)
RHS =  q*Zero * dx

# Form for use in constructing preconditioner matrix
aP = inner(grad(u), grad(v))*dx + p*q*dx

parameters = {
    "ksp_type": "gmres",
    "option" : None,
    "ksp_rtol": 1e-8,
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "schur",
    "pc_fieldsplit_schur_fact_type": "full",
    "pc_fieldsplit_off_diag_use_amat": True,
    "fieldsplit_0_ksp_type": "preonly",
    "fieldsplit_0_pc_type": "lu",
    "fieldsplit_0_pc_factor_mat_solver_package": "mumps",
    "fieldsplit_1_ksp_type": "preonly",
    "fieldsplit_1_pc_type": "lu"
}

# assembling & solving
Problem = LinearVariationalProblem(LHS, RHS, up, aP=aP, bcs = bcs)
Solver = LinearVariationalSolver(Problem, nullspace=nullspace,solver_parameters = parameters)
Solver.solve()
end = time.time()
CGT = end-start

# Showing solution
u, p = up.split()
u.rename("Velocity")
p.rename("Pressure")
File("stokes.pvd").write(u, p)

uexact = Function(V)
uexact.project(u_0)
error2 = np.log(norm(u-uexact))

print("H(div) error")
print(error1)
print("Time take by H(div) solver : "+ str(RinsT) + "seconds")
print("CG error")
print(error2)
print("Time take by H(div) solver : "+ str(CGT) + "seconds")
