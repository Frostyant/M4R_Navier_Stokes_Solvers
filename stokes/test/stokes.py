from firedrake import *
from matplotlib import *
import numpy as np

#Some settings
n = 50 # number of grid points
h = 1/n # "length" of each side
viscosity = 1 #viscosity
c = 20 # works
f = Constant((0,0))
gamma = Constant((100.0))

# Load mesh
mesh = UnitSquareMesh(n, n)

# Define function spaces
V = FunctionSpace(mesh, "BDM", 1)
Q = FunctionSpace(mesh, "DG", 0)
W = V * Q
#defining
x,y= SpatialCoordinate(mesh)
#defining the normal
n = FacetNormal(mesh)

# boundary for penatly
#fbc = as_vector([x^2,-2*y*x])
u_0 = as_vector([0*x**2,0*(-2)*y*x])

# Boundary Conditions #
noslip = Constant((0.0, 0.0))

#Bc1
bc1 = DirichletBC(W.sub(0), u_0, 1) #Can only set Normal Component, here that is u
#Bc2
bc2 = DirichletBC(W.sub(0), u_0, 2)#Can only set Normal Component, here that is v
#Bc3
bc3 = DirichletBC(W.sub(0), u_0, 3)#Can only set Normal Component, here that is u
#Bc4
bc4 = DirichletBC(W.sub(0), u_0, 4)#Can only set Normal Component, here that is v

#boundary conditions
bcs=(bc1,bc2,bc3,bc4)

#pressure
up = Function(W)

# Removing Pressure constant
nullspace = MixedVectorSpaceBasis(
    W, [W.sub(0), VectorSpaceBasis(constant=True)])

# Define variational problem #

#setting up trial and test functions
(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)

#Assembling LHS
L = c/h*inner(v,u_0)*ds
#v0 = as_vector([sin(pi*y),0])
#L = inner(v,v0)*dx

#dealing with viscous term
viscous_byparts1 = inner(grad(u), grad(v))*dx #this is the term over omega from the integration by parts
viscous_byparts2 = 2*inner(avg(outer(v,n)),avg(grad(u)))*dS #this the term over interior surfaces from integration by parts
viscous_symetry = 2*inner(avg(outer(u,n)),avg(grad(v)))*dS #this the term ensures symetry while not changing the continuous equation
viscous_stab = c*1/h*inner(jump(v),jump(u))*dS #stabilizes the equation
viscous_byparts2_ext = (inner(outer(v,n),grad(u)) + inner(outer(u,n),grad(v)))*ds #This deals with boundaries
viscous_ext = c/h*inner(v,u)*ds #this is a penalty term for the boundaries

viscous_term = (
     viscous_byparts1
    - viscous_byparts2
    - viscous_symetry
    + viscous_stab
    - viscous_byparts2_ext
    + viscous_ext #Increasing importance of boundary penalty term
    )# assembles everything

graddiv_term = gamma*div(v)*div(u)*dx

a = (
    viscosity*viscous_term +
    q * div(u) * dx + p * div(v) * dx
    + graddiv_term
    )

#Solving problem #

parameters = {
    "ksp_type": "gmres",
    "ksp_monitor": True,
    "ksp_rtol": 1e-8,
    "pc_type": "fieldsplit",
    "pc_fieldsplit_type": "schur",
    "pc_fieldsplit_schur_fact_type": "full",
    "pc_fieldsplit_off_diag_use_amat": True,
    "fieldsplit_0_ksp_type": "preonly",
    "fieldsplit_0_pc_type": "lu",
    "fieldsplit_1_ksp_type": "preonly",
    "fieldsplit_1_pc_type": "lu"
}

pmass = q*p*dx

aP = viscous_term   + (viscosity + gamma)*pmass +graddiv_term

stokesproblem = LinearVariationalProblem(a,L, up, aP=aP,
                                         bcs=(bc1,bc2,bc3,bc4))

stokessolver = LinearVariationalSolver(stokesproblem,
                                       nullspace=nullspace,
                                       solver_parameters=parameters)

stokessolver.solve()

u, p = up.split()
u.rename("Velocity")
p.rename("Pressure")

 # Plot solution
File("stokes.pvd").write(u, p)
