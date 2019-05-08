from firedrake import *

n = 16
mu = 1

#setting up function spaces and mesh
mesh = UnitSquareMesh(n, n)
x,y= SpatialCoordinate(mesh)
V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)
W = V*Q

#setting up boundary conditions
u_0 = as_vector([cos(y)*exp(x),-sin(y)*exp(x)])
p_0 = Constant(1)*sin(2*pi*x)**2*sin(2*pi*y)**2
p_x = p_0.dx(0)
p_y = p_0.dx(1)
u_x = u_0.dx(0)
u_y = u_0.dx(1)
u_xx = u_x.dx(0)
u_yy = u_x.dx(1)
F_ = as_vector([ - mu*(u_xx[0]+u_yy[0])
            , - mu*(u_xx[1] + u_yy[1])])
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
Zero = Function(Q)
RHS =  inner(v,F) * dx #q*Zero * dx

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

# Showing solution
u, p = up.split()
u.rename("Velocity")
p.rename("Pressure")
File("stokes.pvd").write(u, p)

uexact = Function(V)
uexact.project(u_0)
error = np.log(norm(u-uexact))

print(error)
