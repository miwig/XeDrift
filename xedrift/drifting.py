import numpy as np

def driftRHS(field,drift_velocity,t,x):
    f = field.getValue(x)
    fs = np.linalg.norm(f)
    f = f/fs
    return -f*drift_velocity(fs)
