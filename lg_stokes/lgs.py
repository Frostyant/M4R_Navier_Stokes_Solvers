from firedrake import *
import numpy as np

n = 64
mu = 1

#setting up function spaces and mesh
mesh = UnitSquareMesh(n, n)
x,y= SpatialCoordinate(mesh)
V = VectorFunctionSpace(mesh, "CG", 2)
Q = FunctionSpace(mesh, "CG", 1)
W = V*Q

#setting up boundary conditions
u_0 = as_vector([-sin(2*pi*y)**5*cos(2*pi*y)*sin(2*pi*x)**6,sin(2*pi*x)**5*cos(2*pi*x)*sin(2*pi*y)**6])
p_0 = Constant(10**(-2))*sin(2*pi*x)**3*sin(2*pi*y)**3
p_x = p_0.dx(0)
p_y = p_0.dx(1)
u_x = u_0.dx(0)
u_y = u_0.dx(1)
u_xx = u_x.dx(0)
u_yy = u_y.dx(1)
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
RHS =  inner(v,F) * dx

solver_parameters={"ksp_type": "gmres",
                             "mat_type": "aij",
                             "pc_type": "lu",
                             "pc_factor_mat_solver_type": "mumps"}

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
error = np.log(norm(u-uexact))
perror = np.log(norm(p-pexact))

print(error)
print(perror)

#saving exact values of the error
valfile2 = open("error.txt","w+")
valfile2.write(str(error) + " " + str(perror))
valfile2.close()
