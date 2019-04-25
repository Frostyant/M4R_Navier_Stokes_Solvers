from firedrake import *
from matplotlib import *
import numpy as np
import matplotlib.pyplot as plt

Ns = [2**(n+3) for n in range(5)]
errors = [0]*len(Ns)
ep = [0]*len(Ns)

for it,n in enumerate(Ns):

    #setting up function spaces and mesh
    mesh = UnitSquareMesh(n, n)
    x,y= SpatialCoordinate(mesh)
    V = VectorFunctionSpace(mesh, "CG", 2)
    Q = FunctionSpace(mesh, "CG", 1)
    W = V*Q
    mu = Constant(1)

    u_0 = as_vector([sin(2*pi*y)*cos(2*pi*y)*sin(2*pi*x)**2,-sin(2*pi*x)*cos(2*pi*x)*sin(2*pi*y)**2])
    p_0 = Constant(1)*sin(2*pi*x)**2*sin(2*pi*y)**2
    p_x = p_0.dx(0)
    p_y = p_0.dx(1)
    u_x = u_0.dx(0)
    u_y = u_0.dx(1)
    u_xx = u_x.dx(0)
    u_yy = u_x.dx(1)
    F_ = as_vector([p_x - mu*(u_xx[0]+u_yy[0])
                ,p_y - mu*(u_xx[1] + u_yy[1])])
    F = Function(V)
    F.project(F_)
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
    LHS = inner(grad(u), grad(v)) * dx - div(v) * p * dx + q * div(u) * dx
    RHS =  -inner(v,F) * dx

    # Form for use in constructing preconditioner matrix
    #aP = inner(grad(u), grad(v))*dx + p*q*dx

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
    Problem = LinearVariationalProblem(LHS, RHS, up, bcs = bcs)
    Solver = LinearVariationalSolver(Problem, nullspace=nullspace,solver_parameters = parameters)
    Solver.solve()

    # Showing solution
    u, p = up.split()
    u.rename("Velocity")
    p.rename("Pressure")
    File("stokes.pvd").write(u, p)

    uexact = Function(V)
    uexact.project(u_0)
    pexact = Function(Q)
    pexact.project(p_0)
    errors[it] = norm(u-uexact)
    ep[it] = norm(p-pexact)

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
