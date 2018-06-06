from xedrift.tpc import *
import numpy as np

def outsideTPC(x):
    return x[0] > r_tpc + 0.01 or x[1] < z_tpc - 0.01

def toSurface(RHS,x0,dt=1e-6,t_max = 2e-3):
    x=x0
    t=0
    
    while x[1] < z_liquid and x[1] > z_tpc - 2 and not outsideTPC(x):        
        if x[1] > -2.0:
            dt = 0.2e-6
        
        x=x+dt*RHS(t,tuple(x))
        t=t+dt
        
        if(outsideTPC(x) or t > t_max):
            return np.array([np.NAN, np.NAN])
    
    return np.array([x[0], t * 1e6]) #(µs)

def stream(RHS,x0,dt=1e-6,t_max = 2e-3):
    stream = np.array([*x0,0],ndmin=2)
    x=x0
    t=0

    while x[1] < z_liquid and x[1] > z_tpc - 2 and not outsideTPC(x) and not t>t_max:        
        if x[1] > -2.0:
            dt = 0.2e-6
        
        x=x+dt*RHS(t,tuple(x))
        t=t+dt
        stream = np.append(stream,[[*x,t]],axis=0)
    
    return stream
