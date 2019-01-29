from firedrake import *
from matplotlib import *
import numpy as np

class rinsp:
    """A Navier-Stokes Problem with an efficient pre-build solver using Hdiv"""

    #these are the default parameters
    parameters = {
        "ksp_type": "gmres",
        "ksp_converged_reason": True,
        "ksp_rtol": 1e-6,
        "ksp_max_it": 50,
        "pc_type": "fieldsplit",
        "pc_fieldsplit_type": "schur", #use Schur preconditioner
        "pc_fieldsplit_schur_fact_type": "full", #full preconditioner
        "pc_fieldsplit_off_diag_use_amat": True,
        "fieldsplit_0_ksp_type": "preonly",
        "fieldsplit_0_pc_type": "lu",#use full LU factorization, ilu fails
        "fieldsplit_0_pc_factor_mat_solver_package": "mumps",
        "fieldsplit_1_ksp_type": "preonly",
        "fieldsplit_1_pc_type": "bjacobi",
        "fieldsplit_1_pc_sub_type": "ilu"#use incomplete LU factorization on the submatrix
    }

    c = Constant(20) # works

    def __init__(self, mesh,u_0,W,x,y,z = 0,viscosity = 1,AdvectionSwitchStep = 1,
     gamma = (10**10.0),AverageVelocity = 1,LengthScale = 1,BcIds = 0,dbcIds = 0):
     """ Creats rinsp object

     Keyword arguments:
     mesh -- mesh on which the problem is.
     u_0 -- a function which defines boundary values at ALL boundaries (use conditional if need be).
     W -- FunctionSpace we are working with, program was built with BDM 2 and DG 1.
     x -- x Spatial coordinate.
     y -- y Spatial coordinate.
     z -- z Spatial coordinate, if is default 0 then this is a 2D problem.
     viscosity -- viscosity in problem, default 1.
     AdvectionSwitchStep -- Guess for advection stepsize default 1. If high Reynolds number may want to decrease this for effeciency.
     gamma -- A constant, the higher it is the more effecient the linear solver becomes, default is 10**10.
     AverageVelocity -- Average velocity for system, used for determining Reynolds number, default 1.
     LengthScale -- Length Scale for system, used for determining Reynolds number, default 1.
     BcIds -- Ids for Boundaries on which we apply weak conditions. At default 0 we apply to ALL boundaries. If BcIds and dbcIds are 0 then we assume this is a square.
     dbcIds -- IDs for Boundaries on which we apply strong conditions. At default 0 we just use BcIds instead. If BcIds and dbcIds are 0 then we assume this is a square.
     """

        #setting up basic class attributes
        self.mesh = mesh
        self.u_0 = u_0
        self.viscosity = Constant(viscosity)
        self.AdvectionSwitchStep = AdvectionSwitchStep
        self.gamma = Constant(gamma)
        self.W = W
        self.x = x
        self.y = y
        self.z = 0
        self.AverageVelocity = Constant(AverageVelocity)
        self.R = LengthScale*AverageVelocity/viscosity

        self.bcs =

        gamma = self.gamma
        AverageVelocity = self.AverageVelocity
        viscosity =  self.viscosity
        AdvectionSwitchStep = self.AdvectionSwitchStep

        W = self.W
        #defining
        x,y= self.x,self.y

        #defining the normal
        n = FacetNormal(mesh)

        up = Function(W)

        # Removing Pressure constant
        self.nullspace = MixedVectorSpaceBasis(
            W, [W.sub(0), VectorSpaceBasis(constant=True)])

        # Define variational problem #

        #setting up trial and test functions
        u, p = TrialFunctions(W)
        (self.v, q) = TestFunctions(W)
        v = self.v


        #Assembling LHS
        h = avg(CellVolume(mesh))/FacetArea(mesh)
        if BcIds == 0:
            L = c/(h)*inner(v,u_0)*ds - inner(outer(u_0,n),grad(v))*ds
        else:
            #apply Bcs only to relevant boundaries
            L = c/(h)*inner(v,u_0)*ds(BcIds) - inner(outer(u_0,n),grad(v))*ds(BcIds)

        #Viscous Term parts
        viscous_byparts1 = inner(grad(u), grad(v))*dx #this is the term over omega from the integration by parts
        viscous_byparts2 = 2*inner(avg(outer(v,n)),avg(grad(u)))*dS #this the term over interior surfaces from integration by parts
        viscous_symetry = 2*inner(avg(outer(u,n)),avg(grad(v)))*dS #this the term ensures symetry while not changing the continuous equation
        viscous_stab = c*1/(h)*inner(jump(v),jump(u))*dS #stabilizes the equation
        #Note NatBc turns these terms off, otherwise it is 1
        if BcIds == 0:
            viscous_byparts2_ext = (inner(outer(v,n),grad(u)) + inner(outer(u,n),grad(v)))*ds #This deals with boundaries TOFIX : CONSIDER NON-0 BDARIEs
            viscous_ext =c/(h)*inner(v,u)*ds#this is a penalty term for the boundaries
        else:
            viscous_byparts2_ext = (inner(outer(v,n),grad(u)) + inner(outer(u,n),grad(v)))*ds(BcIDs) #This deals with boundaries TOFIX : CONSIDER NON-0 BDARIEs
            viscous_ext =c/(h)*inner(v,u)*ds(BcIds)#this is a penalty term for the boundaries


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
        self.aP = viscous_term   + (viscosity + gamma)*pmass +graddiv_term

        #Left hand side
        self.F = action(a_bilinear, up) - viscosity*L

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

        self.AdvectionSwitch = Constant(0)

        #Adjusting F with advection term
        self.F += self.AdvectionSwitch*advection_term

        #Adjusting aP, the jacobian, with derivative of advection term
        self.aP += self.AdvectionSwitch*derivative(advection_term, up)

        #Solving problem #

        #Input what we wrote before
        navierstokesproblem = NonlinearVariationalProblem(self.F, up, Jp=self.aP,
                                                          bcs=bcs)
        #Solver
        self.navierstokessolver = NonlinearVariationalSolver(navierstokesproblem,
                                                        nullspace=self.nullspace,
                                                        solver_parameters=parameters)

        #same parameters
        ContinuationParameters = parameters

        #splitting u&p
        self.dupdadvswitch = Function(W)

        #differentiation
        self.RHS = -advection_term

        #replaces all of up in F with dupdadvswitch
        self.LHS = derivative(self.F,up)

        #Input problem
        ContinuationProblem = LinearVariationalProblem(self.LHS,self.RHS,self.dupdadvswitch,aP = self.aP, bcs = bcs)

        #solving
        self.ContinuationSolver = LinearVariationalSolver(ContinuationProblem, nullspace=self.nullspace, solver_parameters = ContinuationParameters)

        self.up = up

    def FullSolve(self,FullOutput = False,Write = True):
        #Fulloutput outputs at EVERY iteration for continuation method
        #Write means that we write down the output at all

        up = self.up
        AdvectionSwitchStep = self.AdvectionSwitchStep

        #This solves the problem
        self.navierstokessolver.solve()

        if Write:
            self.upfile = File("stokes.pvd")

            u, p = up.split()

            u.rename("Velocity")

            p.rename("Pressure")

            self.upfile.write(u, p)
        else:
            FullOutput = False #if we don't write anything then we don't have full output anyway


        #Continuation Method#

        #define
        def ContinuationMethod(self,AdvectionSwitchValue,AdvectionSwitchStep):

            #newton approximation
            self.up += self.dupdadvswitch*(AdvectionSwitchStep)

            self.AdvectionSwitch.assign(AdvectionSwitchValue)
            self.navierstokessolver.solve()

            # Plot solution
            if FullOutput:
                self.upfile.write(u, p)

        #setup variable
        AdvectionSwitchValue = 0

        while AdvectionSwitchValue + AdvectionSwitchStep <= 1:

            try:
                AdvectionSwitchValue += AdvectionSwitchStep
                print(AdvectionSwitchValue)
                ContinuationMethod(self,AdvectionSwitchValue,AdvectionSwitchStep)
                AdvectionSwitchStep = 1.5*AdvectionSwitchStep

                if AdvectionSwitchStep >= (1-AdvectionSwitchValue) and AdvectionSwitchValue < 1:
                    AdvectionSwitchStep = (1-AdvectionSwitchValue)
                    print("Success, solved with full advection")
                else:
                    print("Success, Increasing Step Size")


            except ConvergenceError as ex:
                template = "An Exception of type {0} has occurred. Reducing Step Size."
                print(template.format(type(ex).__name__,ex.args))
                AdvectionSwitchValue -= AdvectionSwitchStep #reset AdvectionSwitchValue
                AdvectionSwitchStep = AdvectionSwitchStep/2
                #IF Advection step is this low the script failed
                if AdvectionSwitchStep <= 10**(-3):
                    print("Too Low Step Size, Solver failed")
                    break
        self.AdvectionSwitch.assign(0)
        self.up = up

        def GetStandardParameters(self):
            #returns standard solve parameters
            return parameters

        def dbc(Ids):
            #sets up dirichelet boundary conditions
            if Ids != 0:
                bcs = (0,)*Ids
            else:
                #if Ids = 0 we assume this is a square where Dirichelet conditions are imposed on all boundaries
                bcs = (0,)*4
                Ids = (1,2,3,4)

            for it,id in Enumerate(Ids):

                bcs[it] = DirichletBC(self.W.sub(0), self.u_0, id)

            self.bcs = bcs




class rinspt(rinsp):
    def __init__(self,ts, mesh,u_0,bcs,W,x,y,t,viscosity = 1,AdvectionSwitchStep = 1,
     gamma = (10**10.0),AverageVelocity = 1,LengthScale = 1,BcIds = 0):
        #initialize standard problem
        #start with t = 0
        self.t = Constant(ts[0])
        self.ts = ts
        self.ub, pb = TrialFunctions(W)
        t.assign(self.t)
        rinsp.__init__(self, mesh,u_0,bcs,W,x,y,viscosity = 1,AdvectionSwitchStep = 1,
         gamma = (10**10.0),AverageVelocity = 1,LengthScale = 1)

    def SolveInTime(self):
        #solves problem in time

        #this stes up the save file for results
        upfile = File("stokes.pvd")

        u, p = self.up.split()

        u.rename("Velocity")

        p.rename("Pressure")

        for it,tval in enumerate(self.ts):

            #updates t
            self.t.assign(tval)

            print(tval)

            if tval != self.ts[0]:

                #splittingsolving u and p for programming purposes (unavoidable)
                u, p = split(self.up)

                self.ub.assign(u)

                DeltaT = float(tval-self.ts[it-1])

                #adding in the finite difference time term
                self.F += inner(u + self.ub,self.v)/DeltaT*dx

                #and its derivative
                self.aP += derivative(inner(u + self.ub,self.v)/DeltaT*dx,self.up)

                #Update problem
                navierstokesproblem = NonlinearVariationalProblem(self.F, self.up, Jp=self.aP,
                                                                  bcs=self.bcs)
                #Update Solver
                self.navierstokessolver = NonlinearVariationalSolver(navierstokesproblem,
                                                                nullspace=self.nullspace,
                                                                solver_parameters=parameters)

                #Update problem
                ContinuationProblem = LinearVariationalProblem(self.LHS,self.RHS,self.dupdadvswitch,aP = self.aP, bcs = self.bcs)

                #Update solver
                self.ContinuationSolver = LinearVariationalSolver(ContinuationProblem, nullspace=self.nullspace, solver_parameters = parameters)

            rinsp.FullSolve(self,FullOutput=False,Write=False)

            u, p = self.up.split()

            upfile.write(u, p,time = tval)
