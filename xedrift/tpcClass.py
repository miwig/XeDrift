import numpy as np


class TPC_X1T:

    def __init__(self):
        self.z_min= -96.8#-96.85    -96.9
        self.z_liquid = 0.25
        self.r_max=47.92

    #@np.vectorize
    def drift_velocity(self,e):
        e=e/100
        p = [-0.03522, 0.03884, -0.000417,2.127e-6,-4.164e-9]
        if(e>165):
            v = 1.5#489
        else:
            v = p[0] + p[1]*e + p[2] * e**2 + p[3]*e**3 + p[4]*e**4

        return v*1e5 #cm/s

    def inside(self,x):
        return x[0] < self.r_max + 0.01 and x[1] > self.z_min - 0.01

    def inside3D(self,x):
        return x[0]**2 + x[1]**2 < (self.r_max + 0.01)**2 and x[2] > self.z_min - 0.01