from xedrift.tpc import *
from scipy.integrate import ode
import numpy as np

def surfaceCondition(t,y):
    if(y[1] > z_liquid):
        return -1
    else:
        return None

def toSurface(RHS,x0, dt=1e-5, t_max=1):
    t0 = 0
    r = ode(RHS).set_integrator('dopri5')
    
    r.set_solout(surfaceCondition)
    r.set_initial_value(x0, t0)
    while r.successful() and r.y[1] < 0.25 and r.t < t_max:
        if r.y[1] > -2.0:
            dt = 0.5e-6
            
        if r.y[1] < z_tpc - 0.1:
            return np.array([np.NAN, np.NAN])
        
        try:              
            r.integrate(r.t+dt)
        except IndexError as e:
            print(e)
            break
            
            
    return np.array([r.y[0], r.t * 1e6]) #(µs)


def stream(RHS,x0, dt=1e-5, t_max=1):
    stream = np.array([*x0,0],ndmin=2)
    t0 = 0
    r = ode(RHS).set_integrator('dopri5')
    r.set_solout(surfaceCondition)
    r.set_initial_value(x0, t0)    
    
    while r.successful() and r.y[1] < z_liquid and r.t < t_max:
        try:
            if r.y[1] > -2.0:
                dt = 0.5e-6
            
            r.integrate(r.t+dt)            
            stream = np.append(stream,[np.append(r.y,[r.t])],axis=0)
        except IndexError as e:
            print(e)
            print(stream)
            break
            
    return stream