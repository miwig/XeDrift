import numpy as np

def driftRHS(field,drift_velocity,t,x):
    """The rhs of the equation for drifting along field lines
    
    Args:
        field: a `xedrift.fields.Field` object representing the field to drift in
        drift_velocity: a function giving the drift velocity in the medium dependent on field strength
        t: time, dummy parameter for algorithms that expect explicitly time-dependent rhs
        x: particle position
    """
    f = field.getValue(x)
    fs = np.linalg.norm(f)
    f = f/fs
    return -f*drift_velocity(fs)

def driftRHS_3D(field,drift_velocity,t,x):
    """The rhs of the equation for drifting along field lines, optimized for 3D space
    
    Args:
        field: a `xedrift.fields.Field` object representing the field to drift in
        drift_velocity: a function giving the drift velocity in the medium dependent on field strength
        t: time, dummy parameter for algorithms that expect explicitly time-dependent rhs
        x: particle position
    """
    f = field.getValue(x)
    fs = np.sqrt(f[0]**2 + f[1]**2 + f[2]**2)
    f = f/fs
    return -f*drift_velocity(fs)