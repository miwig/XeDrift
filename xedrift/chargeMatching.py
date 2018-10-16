from functools import partial
from xedrift.driftingNaive import Drifting2D
from xedrift.driftingNaive3D import Drifting3D
import numpy as np

def volumes2D(simps):
    return np.array([vol2D(simp) for simp in simps])

def rsqint(v0,v1):
    r0,z0 = v0
    r1,z1 = v1

    dr = r1-r0
    dz = z1-z0
    if dr == 0:
        return dz*r0**2

    return dz/dr * (r1**3-r0**3)/3

def vol2D(verts):
    """Calculate volume of solid generated by rotating triangle around z"""
    vs = sorted(verts,key=lambda v:v[1]) #sort by z coordinate
    vb,vm,vt = vs #bottom, middle, top

    return np.abs(rsqint(vb,vm) + rsqint(vm,vt) - rsqint(vb,vt))

from scipy.spatial import ConvexHull
def volumes(simps):
    def volHelper(simp):
        try:
            return ConvexHull(simp).volume
        except Exception:
            #print(simp)
            return -1

    return np.array([volHelper(simp) for simp in simps])

def counts_volumes(grid_real,tri,tri_mask,simpdf):
    dimension = len(grid_real[0])
    simps_real = grid_real[tri.simplices]
    if(dimension==2):
        volumes = volumes2D(simps_real) #special case: r-z space
    else:
        volumes = volumes(simps_real)

    volume_mask = volumes > 0
    if(not np.all(volume_mask)):
        print("All volumes > 0: {}".format(np.all(volume_mask)))


    new_mask = tri_mask.copy()#np.logical_and(tri_mask, volume_mask)

    vol_tpc = np.sum(volumes[new_mask]) #correct?
    #print("TPC volume %: {0:.3f}".format(vol_tpc/(tpc_x1t.r_max**2 * -(tpc_x1t.z_min))))
    count_expected = volumes/vol_tpc * simpdf.events[new_mask].sum() #correct? or use all events?
    count_std = np.sqrt(count_expected) #correct? (or sqrt(simpdf.events)? no! p(D|Q)!
    return count_expected, count_std, volumes, new_mask

def deviations(simpdf,count_expected,count_std,tri_mask):
    return ((simpdf.events - count_expected)/count_std)[tri_mask]

def driftHelp(drifter,x0):
    return drifter.driftReverse(x0)

def log_like_grid(grid_real,triangulation,tri_mask,simpdf):
    count_expected, count_std, volumes, new_mask = counts_volumes(grid_real,triangulation,tri_mask,simpdf)
    devs = deviations(simpdf,count_expected, count_std, new_mask)
    log_l = -0.5*devs**2 - np.log(count_std[new_mask])
    return log_l, devs

def log_like(tpc,superpos,grid_obs,triangulation,tri_mask,simpdf,pool,coeffs,field_override=None):
    if(field_override):
        field = field_override
    else:
        field = superpos.getWithBase(coeffs)

    if(field.dim_space==2):
        drifter = Drifting2D(field,tpc)
    else:
        drifter = Drifting3D(field,tpc)

    drift_f = partial(driftHelp,drifter)
    grid_r = np.array(pool.map(drift_f, grid_obs))

    log_l, devs = log_like_grid(grid_r,triangulation,tri_mask,simpdf)

    return np.sum(log_l), grid_r

def MHStep(state,log_l_f,log_l_prev,scale=0.2,nudgeSingle=True):
    #make new guess
    newstate = state.copy()
    if(nudgeSingle):
        idx = np.random.randint(0,len(state))
        newstate[idx] += np.random.normal(scale=scale)
    else:
        newstate = state + np.random.normal(scale=scale,size=len(state))

    print("New state: {}".format(np.array2string(newstate, formatter={'float_kind':'{0:.3f}'.format})))
    log_l, *meta = log_l_f(newstate)
    print("{:.3f}/{:.3f}".format(log_l,log_l_prev))
    l_ratio = np.exp(log_l - log_l_prev)
    print("L ratio: {:.3f}".format(l_ratio))
    #accept newstate with probability l_ratio
    if(np.random.uniform(0,1) < l_ratio):
        return (newstate, log_l, True, *meta)
    else:
        return (state, log_l_prev, False, *meta)