import numpy as np

def driftRHS(field,drift_velocity,t,x):
    f = field.getValue(x)
    fs = np.linalg.norm(f)
    f = f/fs
    return -f*drift_velocity(fs)

def driftRHS_3D(field,drift_velocity,t,x):
    f = field.getValue(x)
    fs = np.sqrt(f[0]**2 + f[1]**2 + f[2]**2)
    f = f/fs
    return -f*drift_velocity(fs)