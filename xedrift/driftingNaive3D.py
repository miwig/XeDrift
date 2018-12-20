import numpy as np
import xedrift.drifting as drifting
from functools import partial

class Drifting3D:
    def __init__(self,field,tpc):
        self.field = field
        self.tpc = tpc
        self.RHS = partial(drifting.driftRHS_3D,field,tpc.drift_velocity)

    def toSurface(self,x0,dt=1e-6,t_max = 2e-3):
        x=np.copy(x0)
        t=0

        while x[2] < self.tpc.z_liquid and x[2] > self.tpc.z_min - 2 and self.tpc.inside3D(x):
            if x[2] > -2.0:
                dt = 0.2e-6


            dx = dt*self.RHS(t,tuple(x))
            if dx[2] < 0 and x[2] > -0.6:
                break #HACK because drift fields interpolated from low resolution data can cause wrong behavior

            x = x+dx
            t = t+dt

            dx = np.abs(dx)

            if(not self.tpc.inside3D(x) or t > t_max or (dx[0] < 1e-5 and dx[1] < 1e-5 and dx[2] < 1e-5)):#np.allclose(dx,0)):
                return np.array([np.NAN, np.NAN, t*1e6])

        return np.array([x[0], x[1], t * 1e6]) #(µs)
    
    def driftReverse(self,xyt0,dt=1e-6):
        x=np.array([xyt0[0],xyt0[1],0])
        t=xyt0[2]/1e6

        while self.tpc.inside3D(x) and t>0:
            if x[2] < -95.0 or t < 2e-5:
                dt = 0.2e-6

            x=x-dt*self.RHS(t,tuple(x))
            t=t-dt

        return np.array([x[0], x[1], x[2]]) #(µs)

    def stream(self,x0,dt=1e-6,t_max = 2e-3):
        stream = np.array([*x0,0],ndmin=2)
        x=np.copy(x0)
        t=0

        while x[2] < self.tpc.z_liquid and x[2] > self.tpc.z_min - 2 and self.tpc.inside3D(x) and t < t_max:
            if x[2] > -2.0:
                dt = 0.2e-6

            dx=dt*self.RHS(t,tuple(x))
            print(dt, dx)
            x=x+dx
            t=t+dt
            stream = np.append(stream,[[*x,t]],axis=0)

        return stream
