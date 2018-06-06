from xedrift.tpc import *
import matplotlib.pyplot as plt

def drawTPC(ax):
    ax.hlines(z_liquid,colors='black',xmin=0,xmax=r_tpc)
    ax.hlines(z_tpc,colors='black',xmin=0,xmax=r_tpc)
    ax.vlines(r_tpc,colors='black',ymin=z_tpc,ymax=z_liquid)
    ax.hlines(0,colors='black',linestyles=':',xmin=0,xmax=r_tpc)
    
    
def plotStream(stream,**kwargs):
    plt.plot(stream[:,0],stream[:,1],**kwargs)