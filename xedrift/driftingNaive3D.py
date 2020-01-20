import numpy as np
import xedrift.drifting as drifting
from functools import partial

class Drifting3D:
    """A class for drifting particles along electric field lines in a TPC in 3D using the Euler method"""
    def __init__(self,field,tpc):
        """
            Args:
                field: a `xedrift.fields.Field` object representing the field to drift in
                tpc: an object representing the properties of the TPC, i.e. physical dimensions and drift velocity
        """
        self.field = field
        self.tpc = tpc
        self.RHS = partial(drifting.driftRHS_3D,field,tpc.drift_velocity)

    def toSurface(self,x0,dt=1e-6,t_max = 2e-3):
        """Drift a particle to the liquid surface
        
            Args:
                x0: starting position
                dt: time step in seconds
                t_max: maximum drift time in seconds before drifting is aborted
                
            Returns:
                a `numpy.array` containing x, y position at the liquid surface and drift time in µs
        """
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
        """Given the observed x,y-position at the liquid surface and the drift time, calculate the starting point of the particle by reverse drifting
        
            Args:
                xyt0: array-like containing x,y and drift time in seconds
                dt: time step in seconds
            
            Returns:
                a `numpy.array` containing x, y, z for the starting point of the particle
        """
        x=np.array([xyt0[0],xyt0[1],0])
        t=xyt0[2]/1e6

        while self.tpc.inside3D(x) and t>0:
            if x[2] < -95.0 or t < 2e-5:
                dt = 0.2e-6

            x=x-dt*self.RHS(t,tuple(x))
            t=t-dt

        return np.array([x[0], x[1], x[2]]) #(µs)

    def stream(self,x0,dt=1e-6,t_max = 2e-3):
        """Drift a particle to the liquid surface while keeping track of the intermediate positions for visualization"""
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
