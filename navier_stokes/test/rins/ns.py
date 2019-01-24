from firedrake import *
from matplotlib import *
import numpy as np
import rins

#Some settings
c = Constant(20) # works
gamma = Constant((10**10.0))
AverageVelocity = Constant(1)
viscosity = Constant(0.01)
AdvectionSwitchStep = 1

# Load mesh
mesh = Mesh("CylinderInPipe.msh")

problem = rins.rinsp()
