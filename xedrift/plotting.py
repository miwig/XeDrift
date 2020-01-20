import xedrift.tpc
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import numpy as np

def drawTPC(ax=None,tpc=xedrift.tpc,r_squared=False):
    if(ax is None):
        ax = plt.gca()

    if(r_squared):
        r_max = tpc.r_max**2
    else:
        r_max = tpc.r_max
    ax.hlines(tpc.z_liquid,colors='black',xmin=0,xmax=r_max)
    ax.hlines(tpc.z_min,colors='black',xmin=0,xmax=r_max)
    ax.vlines(r_max,colors='black',ymin=tpc.z_min,ymax=tpc.z_liquid)
    ax.hlines(0,colors='black',linestyles=':',xmin=0,xmax=r_max)
    

def plotStream(stream,**kwargs):
    plt.plot(stream[:,0],stream[:,1],**kwargs)


#piecewise functions used to build charge density basis vectors
@np.vectorize
def trapeze(x):
    return np.piecewise(x,[-1/4<x<1/4,
                           1/4<=x<3/4,
                           3/4<=x<5/4],
                           [lambda x: (x+1/4)*2,
                            1,
                            lambda x: 1+(x-3/4)*-2,
                           0])

def ramp_down(x):
    return np.piecewise(x,[-1/4<x,
                           3/4<=x,
                           5/4<=x],
                           [1,
                            lambda x: 1/2 - 2*(x-1),
                            0])

def ramp_up(x):
    return np.piecewise(x,[-1/4<x,
                           1/4<=x],
                           [lambda x: 1/2 + 2*x,
                            1])

def trapeze_or_ramp(x,x0,n_x):
    if(x0==0):
        return ramp_down(x)
    elif(x0==n_x-1):
        return ramp_up(x)
    else:
        return trapeze(x)
    
def trapeze_loop(x,x0,n_x):
    if(x0==0):
        return trapeze(x) + trapeze(x-n_x)
    else:
        return trapeze(x)

def single_charge(n_phi,n_z,z_len,phi0,z0,phi,z):
    """Charge density function for a single charge basis vector on a cylindrical surface
    
    Args:
        n_phi: number of angular subdivisions of the surface
        n_z: number of vertical subdivisions
        z_len: height of the cylinder
        phi0: index of the phi subdivision where the charge is placed
        z0: index of the vertical subdivision where the charge is placed
        phi: angle at which to evaluate the charge density
        z: vertical position at which to evaluate the charge density
    """
    dz = z_len/(n_z)
    dphi = np.pi*2/(n_phi)
    return trapeze_or_ramp(-z/dz - z0,z0,n_z) * trapeze_loop(phi/dphi - phi0 + 0.5,phi0,n_phi)

def charge_dist(coeffs,n_phi,n_z,z_len,phi,z):
    """Returns charge density at phi, z given, built from `single_charge` basis vectors multiplied with coefficients. Slow to evaluate"""
    cs = np.reshape(coeffs,(n_phi,n_z))
    result = 0
    for (iphi, iz), c in np.ndenumerate(cs):
        result += c*single_charge(n_phi,n_z,z_len,iphi,iz,phi,z)

    return result

from functools import lru_cache

@lru_cache(maxsize=16)
def get_base_Qs(n_phi,n_z,z_len):
    """memoized helper so charge_dist_fast doesn't always have to generate its "basis vectors" for charge superpositons"""
    grid_phi = 8*48
    grid_z = 2*48
    PHI, Z = np.meshgrid(np.linspace(0,360,grid_phi,endpoint=False),np.linspace(-z_len,0,grid_z))
    Q_top = single_charge(n_phi,n_z,z_len,0,0,np.radians(PHI),Z)
    Q_middle = single_charge(n_phi,n_z,z_len,0,1,np.radians(PHI),Z)
    Q_bottom = single_charge(n_phi,n_z,z_len,0,n_z-1,np.radians(PHI),Z)
    
    return Q_top, Q_middle, Q_bottom

def charge_dist_fast(coeffs,n_phi,n_z,z_len):
    """Given coefficients, returns a PHI, Z, meshgrid and charge distribution evaluated on that grid, for plotting.
    Faster than evaluating charge_dist on grid because it uses np.roll on "basis vectors" instead"""
    grid_phi = 8*48
    grid_z = 2*48
    PHI, Z = np.meshgrid(np.linspace(0,360,grid_phi,endpoint=False),np.linspace(-z_len,0,grid_z))
    
    roll_phi = int(grid_phi/n_phi)
    roll_z = int(grid_z/n_z)
    
    Q_top, Q_middle, Q_bottom = get_base_Qs(n_phi,n_z,z_len)
    
    def getQ(z):
        if z==0:
            return Q_top
        if z==n_z-1:
            return Q_bottom
        else:
            return np.roll(Q_middle,roll_z * -(z-1),axis=0)
    
    cs = np.reshape(coeffs,(n_phi,n_z))
    
    Q = np.zeros_like(Q_top)
    for (iphi, iz), c in np.ndenumerate(cs):
        Q += c * np.roll(getQ(iz),iphi*roll_phi,axis=1)

    return PHI, Z, Q

def plot_density(PHI,Z,Q,title=None,**kwargs):
    """Plot a charge density on a PHI, Z meshgrid (as returned by `charge_dist_fast`)"""
    ax = kwargs.pop('ax',plt.gca())
    if np.all(Q <= 0):
        lowerl = kwargs.pop('q_min',np.min(Q)-0.02)
        upperl = np.max((np.max(Q)+0.01,0))
        #cmap='viridis_r'
        cmap= kwargs.pop('cmap','viridis_r')
    else:
        syml = np.max(np.abs(Q)+0.01)
        lowerl = -syml
        upperl = syml
        cmap='RdBu_r'
        
    levels = np.linspace(lowerl,upperl,100)
    ticks = np.round(np.linspace(lowerl,upperl,7),2)

    C = ax.contourf(PHI,Z,np.round(Q,2),levels,vmin=lowerl,vmax=upperl,cmap=cmap)

    ax.xaxis.set_major_formatter(tck.FormatStrFormatter('%g°'))
    ax.xaxis.set_major_locator(tck.MultipleLocator(base=45))
    plt.colorbar(C,ax=ax,label=r'Charge density [$\frac{\mathrm{\mu C}}{\mathrm{m}^2}$]',ticks=ticks)

    ax.set_xlabel('$\phi$')
    ax.set_ylabel('$z$ [$\mathrm{cm}$]')
    ax.set_title(title)
