from firedrake import *
from matplotlib import *
import numpy as np

class rinsp:
    """A Navier-Stokes Problem with an efficient pre-build solver using Hdiv"""

    def __init__(self, mesh,u_0,W,x,y,z = 0,viscosity = 1,AdvectionSwitchStep = 1,
    gamma = (10**10.0),AverageVelocity = 1,LengthScale = 1,BcIds = False,DbcIds = False,twoD=True):
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
        DbcIds -- IDs for Boundaries on which we apply strong conditions. At default 0 we just use BcIds instead. If BcIds and dbcIds are 0 then we assume this is a square.
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
        self.BcIds = BcIds
        self.DbcIds = DbcIds
        self.twoD = twoD
        AverageVelocity = self.AverageVelocity
        AdvectionSwitchStep = self.AdvectionSwitchStep
        W = self.W
        x,y= self.x,self.y
        self.up = Function(W)
        up = self.up
        #these are the default solver parameters
        self.parameters = {
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
        #setting up dirichelet boundary conditions
        if(DbcIds != False):
            #using DbcIds
            self.dbc(DbcIds)
        else:
            #using BcIDs, which are the IDs used for weak boundaries
            self.dbc(BcIds)
        n = FacetNormal(mesh)
        # Removing Pressure constant
        self.nullspace = MixedVectorSpaceBasis(
            W, [W.sub(0), VectorSpaceBasis(constant=True)])

        # Define variational problem #
        u, p = TrialFunctions(W)
        (self.v, self.q) = TestFunctions(W)
        v = self.v

        #These terms create the viscous and numerical stability part of the equation
        viscous_term,L = self.GetViscousTerm(u,p)
        a_bilinear,graddiv_term = self.GetBilinear(u,p,viscous_term)
        self.aP = self.GetApV(u,p,viscous_term,graddiv_term)
        self.F = action(a_bilinear, self.up) - self.viscosity*L

        #These terms are the advective parts of the equation
        advection_term = self.GetAdvectionTerm(self.up)
        self.AdvectionSwitch = Constant(0) #initially we neglect advection
        self.F += self.AdvectionSwitch*advection_term
        self.aP += self.AdvectionSwitch*derivative(advection_term, self.up)

        #Creating Solvers #

        #Input what we wrote before
        navierstokesproblem = NonlinearVariationalProblem(self.F, self.up, Jp=self.aP,
                                                          bcs=self.bcs)
        #Solver
        self.navierstokessolver = NonlinearVariationalSolver(navierstokesproblem,
                                                        nullspace=self.nullspace,
                                                        solver_parameters=self.parameters)
        self.dupdadvswitch = Function(W)
        self.RHS = -advection_term
        self.LHS = derivative(self.F,self.up)


        #Input problem
        ContinuationProblem = LinearVariationalProblem(self.LHS,self.RHS,self.dupdadvswitch,aP = self.aP, bcs = self.bcs)

        #solving
        self.ContinuationSolver = LinearVariationalSolver(ContinuationProblem, nullspace=self.nullspace, solver_parameters = self.parameters)

        #newton approximation
        self.up += self.dupdadvswitch*(AdvectionSwitchStep)

    def FullSolve(self,FullOutput = False,Write = True):
        #Fulloutput outputs at EVERY iteration for continuation method
        #Write means that we write down the output at all

        AdvectionSwitchStep = self.AdvectionSwitchStep

        #This solves the problem
        self.navierstokessolver.solve()

        if Write:
            self.upfile = File("stokes.pvd")

            u, p = self.up.split()

            u.rename("Velocity")

            p.rename("Pressure")

            self.upfile.write(u, p)
        else:
            FullOutput = False #if we don't write anything then we don't have full output anyway


        #Continuation Method#
        def ContinuationMethod(self,AdvectionSwitchValue,AdvectionSwitchStep):

            self.AdvectionSwitch.assign(AdvectionSwitchValue)
            self.ContinuationSolver.solve()

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
                    print("Success, solving with full advection")
                elif AdvectionSwitchValue + AdvectionSwitchStep <= 1:
                    print("Success, increasing step size")
                else:
                    print("Success, solve complete")


            except ConvergenceError as ex:
                template = "An Exception of type {0} has occurred. Reducing Step Size."
                print(template.format(type(ex).__name__,ex.args))
                AdvectionSwitchValue -= AdvectionSwitchStep #reset AdvectionSwitchValue
                AdvectionSwitchStep = AdvectionSwitchStep/2
                #IF Advection step is this low the script failed
                if AdvectionSwitchStep <= 10**(-3):
                    print("Too low step size, solver failed")
                    break
        self.AdvectionSwitch.assign(0)

    def dbc(self,Ids):
        #sets up dirichelet boundary conditions

        if isinstance(Ids,int):
            #this is just to avoid uneccessary bugs, can input integer instead of tuple
            bcs = [DirichletBC(self.W.sub(0), self.u_0, Ids)]
        else:
            if Ids != False:
                bcs = [0,]*len(Ids)
            else:
                #if Ids = 0 we assume this is a square where Dirichelet conditions are imposed on all boundaries
                bcs = [0,]*4
                Ids = (1,2,3,4)

            for it,id in enumerate(Ids):
                bcs[it] = DirichletBC(self.W.sub(0), self.u_0, id)

        self.bcs = tuple(bcs)

    def GetAdvectionTerm(self,up):

        n = FacetNormal(self.mesh)
        #splitting u and p for programming purposes (unavoidable)
        u, p = split(up)

        #Re-Defining functions for use in Advection term
        if self.twoD:
            curl = lambda phi: as_vector([-phi.dx(1), phi.dx(0)])
            cross = lambda u, w: u[0]*w[1]-u[1]*w[0]
            perp = lambda n, phi: as_vector([n[1]*phi, -n[0]*phi])
        else:
            perp = cross

        #Defining upwind and U_upwind for us in advection
        Upwind = 0.5*(sign(dot(u, n))+1)
        U_upwind = Upwind('+')*u('+') + Upwind('-')*u('-')

        #Assembling Advection Term
        adv_byparts1 = inner(u, curl(cross(u, self.v)))*dx #This is the term from integration by parts of double curl
        adv_byparts2 = inner(U_upwind, 2*avg( perp(n, cross(u, self.v))))*dS #Second term over surface
        adv_grad = 0.5*div(self.v)*inner(u,u)*dx #This is the term due to the gradient of u^2
        adv_bdc1 = inner(self.u_0,perp(n,cross(self.u_0,self.v)))*ds #boundary version of adv_byparts2
        adv_bdc2 = 1/2*inner(inner(self.u_0,self.u_0)*self.v,n)*ds #boundary term from u^2 when it is non-0
        advection_term = (
            adv_byparts1
            - adv_byparts2
            - adv_grad
            - adv_bdc1
            + adv_bdc2
        )

        return advection_term

    def GetViscousTerm(self,u,p):
        """Gets Main Viscous Terms
        Keyword arguments:
        u -- velocity
        p -- pressure

        Outputs :
        viscous_term -- RHS Viscous terms (ie they depend on up)
        L -- LHS Viscous Terms (They do not depend on up)
        """
        c = Constant(20)
        n= FacetNormal(self.mesh)
        h = avg(CellVolume(self.mesh))/FacetArea(self.mesh)
        if self.BcIds == False:
            L = c/(h)*inner(self.v,self.u_0)*ds - inner(outer(self.u_0,n),grad(v))*ds
        else:
            #apply Bcs only to relevant boundaries
            L = c/(h)*inner(self.v,self.u_0)*ds(self.BcIds) - inner(outer(self.u_0,n),grad(self.v))*ds(self.BcIds)

        #Viscous Term parts
        viscous_byparts1 = inner(grad(u), grad(self.v))*dx #this is the term over omega from the integration by parts
        viscous_byparts2 = 2*inner(avg(outer(self.v,n)),avg(grad(u)))*dS #this the term over interior surfaces from integration by parts
        viscous_symetry = 2*inner(avg(outer(u,n)),avg(grad(self.v)))*dS #this the term ensures symetry while not changing the continuous equation
        viscous_stab = c*1/(h)*inner(jump(self.v),jump(u))*dS #stabilizes the equation
        #Note NatBc turns these terms off, otherwise it is 1
        if self.BcIds == False:
            viscous_byparts2_ext = (inner(outer(self.v,n),grad(u)) + inner(outer(u,n),grad(self.v)))*ds #This deals with boundaries TOFIX : CONSIDER NON-0 BDARIEs
            viscous_ext =c/(h)*inner(self.v,u)*ds#this is a penalty term for the boundaries
        else:
            viscous_byparts2_ext = (inner(outer(self.v,n),grad(u)) + inner(outer(u,n),grad(self.v)))*ds(self.BcIds) #This deals with boundaries TOFIX : CONSIDER NON-0 BDARIEs
            viscous_ext =c/(h)*inner(self.v,u)*ds(self.BcIds)#this is a penalty term for the boundaries
        #Assembling Viscous Term
        viscous_term = self.viscosity*(
            viscous_byparts1
            - viscous_byparts2
            - viscous_symetry
            + viscous_stab
            - viscous_byparts2_ext
            + viscous_ext
        )
        return viscous_term,L

    def GetBilinear(self,u,p,viscous_term):

        #Setting up bilenar form
        graddiv_term = self.gamma*div(self.v)*div(u)*dx
        a_bilinear = (
            viscous_term +
            self.q * div(u) * dx - p * div(self.v) * dx
            + graddiv_term
        )
        return a_bilinear,graddiv_term

    def GetApV(self,u,p,viscous_term,graddiv_term):
        pmass = self.q*p*dx
        return viscous_term + (self.viscosity + self.gamma)*pmass + graddiv_term



class rinspt(rinsp):
    def __init__(self, mesh,u_0,W,x,y,t,viscosity = 1,AdvectionSwitchStep = 1,
     gamma = (10**10.0),AverageVelocity = 1,LengthScale = 1,BcIds = False,DbcIds = False):
        #initialize standard problem
        self.t=t
        #boundary ids
        self.BcIds = BcIds
        self.DbcIds = DbcIds
        #TimeSwitch for time terms
        self.TimeSwitch = Constant(0)
        rinsp.__init__(self, mesh,u_0,W,x,y,viscosity = viscosity,AdvectionSwitchStep = AdvectionSwitchStep,
         gamma = gamma,AverageVelocity = AverageVelocity,LengthScale = LengthScale,BcIds = BcIds,DbcIds = DbcIds)


        #Adding Time terms#
        #defining upb to store prior value
        self.upb = Function(self.W)
        self.DeltaT = Constant(1)

        #programmtically required
        ub,pb = split(self.upb)
        u, p = split(self.up)

        #adding in the finite difference time term
        self.F += self.TimeSwitch*inner(u + ub,self.v)/self.DeltaT*dx
        #and its derivative
        self.aP += self.TimeSwitch*derivative(inner(u + ub,self.v)/self.DeltaT*dx,self.up)

        #Update problem
        navierstokesproblem = NonlinearVariationalProblem(self.F, self.up, Jp=self.aP,
                                                           bcs=self.bcs)
        #Update Solver
        self.navierstokessolver = NonlinearVariationalSolver(navierstokesproblem,
                                                        nullspace=self.nullspace,
                                                        solver_parameters=self.parameters)
        #Update problem
        ContinuationProblem = LinearVariationalProblem(self.LHS,self.RHS,self.dupdadvswitch,aP = self.aP, bcs = self.bcs)

        #Update solver
        self.ContinuationSolver = LinearVariationalSolver(ContinuationProblem, nullspace=self.nullspace, solver_parameters = self.parameters)

        #setup Picards Solver
        self.PicardIterationSetup()

    def SolveInTime(self,ts,precise = False,PicIt=2):
        """Solves Probem in time using Picard and, if precise = True, Newton in addition
        Keyword arguments:
        precise -- If true will use Newton after Picards iterations at each timestep
        PicIt -- Number of Picards Iteration
        """
        upfile = File("stokes.pvd")
        u, p = self.up.split()
        u.rename("Velocity")
        p.rename("Pressure")

        #first time step
        self.upb.assign(self.up)
        self.t.assign(ts[0])
        rinsp.FullSolve(self,FullOutput=False,Write=False)
        #splitting u and p for programming purposes (unavoidable)
        u, p = self.up.split()
        upfile.write(u, p,time = ts[0])
        ts = np.delete(ts,0)

        for it,tval in enumerate(ts):
            #For coding purposes need to use split(up)
            self.t.assign(tval)
            self.upb.assign(self.up)
            self.DeltaT.assign(float(tval-ts[it-1]))
            print(tval)
            self.PicardsSolver.solve()
            if precise:
                self.TimeSwitch.assign(1)
                rinsp.FullSolve(self,FullOutput=False,Write=False)
                self.TimeSwitch.assign(0)

            #For coding purposes need to use up.split here
            u, p = self.up.split()
            upfile.write(u, p,time = tval)

    def PicardIterationSetup(self,MidNotB = False):
        """Does Picards iterations on the navier stokes solution
        Keyword arguments:
        PicIt -- Number of Picards Iteration
        MidNotB -- If True use Midpoint rule, otherwise use Backwards Euler
        """
        if not MidNotB:
            #We are using advection term from previous step (nonlinear term)
            advection_term = self.GetAdvectionTerm(self.upb)
            u, p = TrialFunctions(self.W)
            viscous_term,L = self.GetViscousTerm(u,p)
            a_bilinear,graddiv_term = self.GetBilinear(u,p,viscous_term)
            self.AdvectionSwitch = Constant(0) #initially we neglect advection
            PicardsProblem = LinearVariationalProblem(a_bilinear,L + advection_term, self.up,
                                                        aP=self.GetApV(u,p,viscous_term,graddiv_term), bcs=self.bcs)
            self.PicardsSolver = LinearVariationalSolver(PicardsProblem, nullspace=self.nullspace, solver_parameters = self.parameters)
