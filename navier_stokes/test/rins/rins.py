from firedrake import *
from matplotlib import *
import numpy as np

class rinsp:
    """R-independent navier-stokes problem"""

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

    def __init__(self, mesh,u_0,bcs,W,x,y,viscosity = 1,AdvectionSwitchStep = 1,
     gamma = (10**10.0),AverageVelocity = 1,LengthScale = 1,BcIds = 0):
        self.mesh = mesh
        self.u_0 = u_0
        self.bcs = bcs
        self.viscosity = Constant(viscosity)
        self.AdvectionSwitchStep = AdvectionSwitchStep
        self.gamma = Constant(gamma)
        self.W = W
        self.x = x
        self.y = y
        self.AverageVelocity = Constant(AverageVelocity)
        self.R = LengthScale*AverageVelocity/viscosity

        c = Constant(20) # works
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
            upfile = File("stokes.pvd")

            u, p = up.split()

            u.rename("Velocity")

            p.rename("Pressure")

            upfile.write(u, p)
        else:
            FullOutput = False #if we don't write anything then we don't have full output anyway


        #Continuation Method#

        #define
        def ContinuationMethod(self,AdvectionSwitchValue,AdvectionSwitchStep):

            global upfile

            #newton approximation
            self.up += self.dupdadvswitch*(AdvectionSwitchStep)

            self.AdvectionSwitch.assign(AdvectionSwitchValue)
            self.navierstokessolver.solve()

            # Plot solution
            if FullOutput:
                upfile.write(u, p)

        #setup variable
        AdvectionSwitchValue = 0

        while AdvectionSwitchValue + AdvectionSwitchStep <= 1:

            try:
                AdvectionSwitchValue += AdvectionSwitchStep
                print(AdvectionSwitchValue)
                ContinuationMethod(self,AdvectionSwitchValue,AdvectionSwitchStep)
                AdvectionSwitchStep = 1.5*AdvectionSwitchStep
                print("Success, Increasing Step Size")
                if AdvectionSwitchStep >= (1-AdvectionSwitchValue) and AdvectionSwitchValue < 1:
                    AdvectionSwitchStep = (1-AdvectionSwitchValue)


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
            return parameters



class rinspt(rinsp):
    def __init__(self,ts, mesh,u_0,bcs,W,x,y,t,viscosity = 1,AdvectionSwitchStep = 1,
     gamma = (10**10.0),AverageVelocity = 1,LengthScale = 1,BcIds = 0):
        #initialize standard problem
        #start with t = 0
        self.t = Constant(ts[0])
        self.ts = ts
        t.assign(self.t)
        rinsp.__init__(self, mesh,u_0,bcs,W,x,y,viscosity = 1,AdvectionSwitchStep = 1,
         gamma = (10**10.0),AverageVelocity = 1,LengthScale = 1)

    def SolveInTime(self):
        #solves problem in time
        for it,tval in enumerate(self.ts):

            self.t.assign(tval)

            print(tval)

            upfile = File("stokes.pvd")

            u, p = self.up.split()

            u.rename("Velocity")

            p.rename("Pressure")

            upfile.write(u, p)

            if tval != self.ts[0]:

                #splittingsolving u and p for programming purposes (unavoidable)
                u, p = split(self.up)

                ub = u

                DeltaT = float(tval-self.ts[it-1])

                #adding in the finite difference time term
                self.F += inner(u + ub,self.v)/DeltaT*dx

                #and its derivative
                self.aP += derivative(inner(u + ub,self.v)/DeltaT*dx,self.up)

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

            rinsp.FullSolve(self)
            
            u, p = self.up.split()

            u.rename("Velocity")

            p.rename("Pressure")

            upfile.write(u, p)
