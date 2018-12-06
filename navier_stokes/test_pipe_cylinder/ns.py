from firedrake import *
from matplotlib import *
import numpy as np

#Some settings
c = Constant(20) # works
f = Constant((1,0))
gamma = Constant((100.0))
AverageVelocity = Constant(1)
MinimalViscosity = 1
MaximalViscosity = 101
ViscosityStepsize = 10
viscosities = range(MinimalViscosity,MaximalViscosity,ViscosityStepsize)

# Load mesh
mesh = Mesh("CylinderInPipe.msh")

# Define function spaces
V = FunctionSpace(mesh, "BDM", 2)
Q = FunctionSpace(mesh, "DG", 1)
W = V * Q
#defining
x,y= SpatialCoordinate(mesh)
#defining the normal
n = FacetNormal(mesh)

# boundary function, these are assumed to not change during iteration
u_0 = as_vector([conditional(x <0.1,AverageVelocity*sin(pi*y)**2,0.) + conditional(x > 0.9,AverageVelocity*sin(pi*y)**2,0.),0])
p_0 = 0

for viscosity in viscosities:
    #Quick Explanation: We cannot solve for high reynolds number, so instead we solve for low viscosity and then gradually increaseself.
    #We then use the u estimated from the prior step as a guess for the next one so that the solver converges.
    #In theory firedrake should automatically use the prior u values as a guess since they are stored in the variable which generates the new test fct.
    
    #Bc1
    bc1 = DirichletBC(W.sub(0), u_0, 1) #Can only set Normal Component, here that is u left bdary
    bc1p = DirichletBC(W.sub(1), p_0, 1)

    #Bc2
    bc2 = DirichletBC(W.sub(0), u_0, 2)#Can only set Normal Component, here that is u right bdary

    #Bc3
    bc3 = DirichletBC(W.sub(1), p_0, 3)#Can only set Normal Component, here that is v bottom bdary

    #Bc4
    bc4 = DirichletBC(W.sub(0), u_0, 4)#Can only set Normal Component, here that is v top bdary

    #Bc5, From cylinder
    bc5 = DirichletBC(W.sub(0), u_0, 5)

    #boundary conditions
    bcs=(bc1,bc1p,bc2,bc3,bc4,bc5)


    up = Function(W)

    # Removing Pressure constant
    nullspace = MixedVectorSpaceBasis(
        W, [W.sub(0), VectorSpaceBasis(constant=True)])

    # Define variational problem #

    #setting up trial and test functions
    u, p = TrialFunctions(W)
    (v, q) = TestFunctions(W)

    #Assembling LHS
    h = avg(CellVolume(mesh))/FacetArea(mesh)
    L = c/(h)*inner(v,u_0)*ds - inner(outer(u_0,n),grad(v))*ds

    #Viscous Term parts
    viscous_byparts1 = inner(grad(u), grad(v))*dx #this is the term over omega from the integration by parts
    viscous_byparts2 = 2*inner(avg(outer(v,n)),avg(grad(u)))*dS #this the term over interior surfaces from integration by parts
    viscous_symetry = 2*inner(avg(outer(u,n)),avg(grad(v)))*dS #this the term ensures symetry while not changing the continuous equation
    viscous_stab = c*1/(h)*inner(jump(v),jump(u))*dS #stabilizes the equation
    #Note NatBc turns these terms off, otherwise it is 1
    viscous_byparts2_ext = (inner(outer(v,n),grad(u)) + inner(outer(u,n),grad(v)))*ds #This deals with boundaries TOFIX : CONSIDER NON-0 BDARIEs
    viscous_ext =c/(h)*inner(v,u)*ds #this is a penalty term for the boundaries

    #Assembling Viscous Term
    viscous_term = viscosity*(
        viscous_byparts1
        - viscous_byparts2
        - viscous_symetry
        + viscous_stab
        - viscous_byparts2_ext
        + viscous_ext
        )

    #Setting up bilenar form
    graddiv_term = gamma*div(v)*div(u)*dx

    a_bilinear = (
        viscous_term +
        q * div(u) * dx - p * div(v) * dx
        + graddiv_term
        )

    pmass = q*p*dx

    #Jacobian
    aP = viscous_term   + (viscosity + gamma)*pmass +graddiv_term

    #Left hand side
    F = action(a_bilinear, up) - L

    #splitting u and p for programming purposes (unavoidable)
    u, p = split(up)

    #Re-Defining functions for use in Advection term
    twoD = True
    if twoD:
        curl = lambda phi: as_vector([-phi.dx(1), phi.dx(0)])
        cross = lambda u, w: u[0]*w[1]-u[1]*w[0]
        perp = lambda n, phi: as_vector([n[1]*phi, -n[0]*phi])
    else:
        perp = cross

    #Defining upwind and U_upwind for us in advection
    Upwind = 0.5*(sign(dot(u, n))+1)
    U_upwind = Upwind('+')*u('+') + Upwind('-')*u('-')

    #Assembling Advection Term
    adv_byparts1 = inner(u, curl(cross(u, v)))*dx #This is the term from integration by parts of double curl
    adv_byparts2 = inner(U_upwind, 2*avg( perp(n, cross(u, v))))*dS #Second term over surface
    adv_grad = 0.5*div(v)*inner(u,u)*dx #This is the term due to the gradient of u^2
    adv_bdc1 = inner(u_0,perp(n,cross(u_0,v)))*ds #boundary version of adv_byparts2
    adv_bdc2 = 1/2*inner(inner(u_0,u_0)*v,n)*ds #boundary term from u^2 when it is non-0
    advection_term = (
        adv_byparts1
        - adv_byparts2
        - adv_grad
        - adv_bdc1
        + adv_bdc2
        )

    #Adjusting F with advection term
    F += advection_term

    #Adjusting aP, the jacobian, with derivative of advection term
    aP += derivative(advection_term, up)

    #Solving problem #
    parameters = {
        "ksp_type": "gmres",
        "ksp_monitor": True,
        "ksp_rtol": 1e-8,
        "pc_type": "fieldsplit",
        "pc_fieldsplit_type": "schur", #use Schur preconditioner
        "pc_fieldsplit_schur_fact_type": "full", #full preconditioner
        "pc_fieldsplit_off_diag_use_amat": True,
        "fieldsplit_0_ksp_type": "preonly",
        "fieldsplit_0_pc_type": "lu",#use full LU factorization, ilu fails
        "fieldsplit_1_ksp_type": "preonly",
        "fieldsplit_1_pc_type": "bjacobi",
        "fieldsplit_1_pc_sub_type": "ilu"#use incomplete LU factorization on the submatrix
    }


    #Input what we wrote before
    navierstokesproblem = NonlinearVariationalProblem(F, up, Jp=aP,
                                                bcs=bcs)
    #Solver
    navierstokessolver = NonlinearVariationalSolver(navierstokesproblem,
                                              nullspace=nullspace,
                                              solver_parameters=parameters)

    #This solves the problem
    navierstokessolver.solve()

    #stores u and p values separetly
    u, p = up.split()

u.rename("Velocity")
p.rename("Pressure")

 # Plot solution
File("stokes.pvd").write(u, p)

 # Plot solution
plot(p)
plot(u)
File("stokes.pvd").write(u, p)
