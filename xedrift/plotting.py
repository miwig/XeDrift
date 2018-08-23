import xedrift.tpc
import matplotlib.pyplot as plt

def drawTPC(ax=None,tpc=xedrift.tpc):
    if(ax is None):
        ax = plt.gca()
    ax.hlines(tpc.z_liquid,colors='black',xmin=0,xmax=tpc.r_max)
    ax.hlines(tpc.z_min,colors='black',xmin=0,xmax=tpc.r_max)
    ax.vlines(tpc.r_max,colors='black',ymin=tpc.z_min,ymax=tpc.z_liquid)
    ax.hlines(0,colors='black',linestyles=':',xmin=0,xmax=tpc.r_max)
    
    
def plotStream(stream,**kwargs):
    plt.plot(stream[:,0],stream[:,1],**kwargs)