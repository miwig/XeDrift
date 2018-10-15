import numpy as np
from xedrift.fields import Field

def rot(X,Y,phi):
    Xr =  np.cos(phi)*X - np.sin(phi)*Y
    Yr =  np.sin(phi)*X + np.cos(phi)*Y
    return Xr, Yr

def rotateField(field, phi):
    """Rotate field around z-axis by angle phi"""
    X,Y,Z = np.meshgrid(*field.grid,indexing='ij')
    Xr, Yr = rot(X,Y,phi) #rotate grid

    fv = field.getValue((Xr,Yr,Z)) #get field values on rotated grid
    fX, fY, fZ = fv[:,:,:,0], fv[:,:,:,1], fv[:,:,:,2]
    fXr, fYr = rot(fX,fY,-phi) #rotate field values back to normal coordinate system
    fvr = np.stack((fXr,fYr,fZ),axis=-1)

    components=sorted(field.components,key=lambda k:field.components[k])    

    field_rot = Field(gridspec=(field.grid,fvr,components))
    return field_rot

import time
import os
import gc

def rotateFile(path,steps,start=1,**kwargs):
    if(not ".txt" in path):
        raise ValueError

    stop = kwargs.get('stop',steps)

    field = Field(path)
    print("Processing {} from {} to {}".format(path,start,stop-1))
    for i in range(start,stop):
        newpath = path.replace(".txt","_rot_{}.txt".format(i))
        if(os.path.exists(newpath) or os.path.exists(newpath+'.npz')):
            print("Found {}, skipping!".format(newpath))
            continue
        print("Processing {}".format(newpath))
        start = time.time()
        phi = np.radians(i*360/steps)
        field_r = rotateField(field,phi)

        field_r.save(newpath)
        del field_r
        gc.collect()
        end = time.time()
        print("Saved {} in {:.2f} seconds".format(newpath,end-start))