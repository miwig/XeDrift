import numpy as np

z_min= -96.8#-96.85    -96.9
z_liquid = 0.25
r_max=47.92

#drift velocity in SR1: 1.335 * um / ns
drift_v = 1.335 * 1000**2 #mm/s

@np.vectorize
def drift_velocity(e):
    e=e/100
    p = [-0.03522, 0.03884, -0.000417,2.127e-6,-4.164e-9]
    if(e>165):
        v = 1.5#489
    else:
        v = p[0] + p[1]*e + p[2] * e**2 + p[3]*e**3 + p[4]*e**4

    return v*1e5 #cm/s

def inside(x):
    return x[0] < r_max + 0.01 and x[1] > z_min - 0.01