from xedrift.tpc import *
import numpy as np

def outsideTPC(x):
    return x[0]**2 + x[1]**2 > (r_tpc + 0.01)**2 or x[2] < z_tpc - 0.01

def toSurface(RHS,x0,dt=1e-6,t_max = 2e-3):
    x=x0
    t=0
    
    while x[2] < z_liquid and x[2] > z_tpc - 2 and not outsideTPC(x):        
        if x[2] > -2.0:
            dt = 0.2e-6
        
        x=x+dt*RHS(t,tuple(x))
        t=t+dt
        
        if(outsideTPC(x) or t > t_max):
            return np.array([np.NAN, np.NAN, t*1e6])
    
    return np.array([x[0],x[1], t * 1e6]) #(µs)

def stream(RHS,x0,dt=1e-6,t_max = 2e-3):
    stream = np.array([*x0,0],ndmin=2)
    x=x0
    t=0

    while x[2] < z_liquid and x[2] > z_tpc - 2 and not outsideTPC(x) and not t>t_max:        
        if x[2] > -2.0:
            dt = 0.2e-6
        
        x=x+dt*RHS(t,tuple(x))
        t=t+dt
        stream = np.append(stream,[[*x,t]],axis=0)
    
    return stream
