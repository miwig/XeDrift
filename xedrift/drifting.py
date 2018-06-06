from scipy.integrate import ode
from xedrift.tpc import *
import numpy as np
import xedrift.driftingNaive

@np.vectorize
def drift_velocity(e):
    e=e/100
    p = [-0.03522, 0.03884, -0.000417,2.127e-6,-4.164e-9]
    if(e>165):
        v = 1.5#489
    else:
        v = p[0] + p[1]*e + p[2] * e**2 + p[3]*e**3 + p[4]*e**4
        
    return v*1e5 #cm/s

def driftRHS(field,t,x): 
    f = field.getValue(x)
    fs = np.linalg.norm(f)
    f = f/fs
    return -f*drift_velocity(fs)


from functools import partial
def getFunctions(field,module=xedrift.driftingNaive):
    RHS = partial(driftRHS,field)
    driftToSurface = partial(module.toSurface,RHS)
    driftStream = partial(module.stream,RHS)
    
    return driftToSurface, driftStream