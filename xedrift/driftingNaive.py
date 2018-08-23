import numpy as np
import xedrift.drifting as drifting
from functools import partial

class Drifting2D:
    def __init__(self,field,tpc):
        self.field = field
        self.tpc = tpc
        self.RHS = partial(drifting.driftRHS,field,tpc.drift_velocity)

    def toSurface(self,x0,dt=1e-6,t_max = 2e-3):
        x=x0
        t=0

        while x[1] < self.tpc.z_liquid and x[1] > self.tpc.z_min - 2 and self.tpc.inside(x):
            if x[1] > -2.0:
                dt = 0.2e-6


            dx = dt*self.RHS(t,tuple(x))
            if dx[1] < 0 and x[1] > -0.6:
                break #HACK because drift fields interpolated from low resolution data can cause wrong behavior

            x=x+dx
            t=t+dt

            if(not self.tpc.inside(x) or t > t_max):
                return np.array([np.NAN, t*1e6])

        return np.array([x[0], t * 1e6]) #(µs)

    def driftReverse(self,rt0,dt=1e-6):
        x=np.array([rt0[0],self.tpc.z_liquid - 0.0])#z_liquid
        t=rt0[1]/1e6

        while self.tpc.inside(x) and t>0:
            if x[1] > -2.0 or t < 5e-5:
                dt = 0.2e-6

            x=x-dt*self.RHS(t,tuple(x))
            t=t-dt

        return np.array([x[0], x[1]]) #(µs)

    def stream(self,x0,dt=1e-6,t_max = 2e-3):
        stream = np.array([*x0,0],ndmin=2)
        x=x0
        t=0

        while x[1] < self.tpc.z_liquid and x[1] > self.tpc.z_min - 2 and self.tpc.inside(x) and not t>t_max:
            if x[1] > -2.0:
                dt = 0.2e-6

            x=x+dt*self.RHS(t,tuple(x))
            t=t+dt
            stream = np.append(stream,[[*x,t]],axis=0)

        return stream