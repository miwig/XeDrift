import numpy as np
import re
from scipy.interpolate import RegularGridInterpolator
import copy

class Field:
    def _set_component(self,name,data):
        setattr(self,name,data)
        self.components[name] = data
        
    def _read_data(self,file):
        attr = file.readline()
        reg = r'\.(.+?)\s'
        attr = re.search(reg,attr).group(1)
        
        numrows = self.shape[1]
        #np.genfromtxt is considerate and reads only the lines it needs to, leaving the file handle intact
        data = np.transpose(np.genfromtxt(file,max_rows=numrows)) 
        
        if not np.shape(data) == self.shape:
            raise Exception("invalid file")
            
        self._set_component(attr,data)
        
        
    def __init__(self,filename=None):
        self.components = {}
        self.interpolator = {}
        
        file = open(filename,'r')
        hasGrid = False

        while True:
            line = file.readline()
            if not line:
                break

            #print(line)
            if not hasGrid and '% Grid' in line:
                self.rs = np.genfromtxt(file,max_rows=1)/10 #convert mm to cm
                self.zs = np.genfromtxt(file,max_rows=1)/10
                self.shape = (len(self.rs),len(self.zs))
                continue

            if '% Data' in line:
                self._read_data(file)
                continue
        
        self._setupInterpolator()
                
    #TODO: generalize for 3d        
    def _setupInterpolator(self):             
        self.interpolator = RegularGridInterpolator((self.rs,self.zs),np.dstack((self.Er,self.Ez)))
             
    def getValue(self,pos):
        try:
            return self.interpolator(pos)
        except ValueError:
            print("Error at {}".format(pos))
            return np.zeros_like(pos)

    def __add__(self, other):
        if not isinstance(other,Field):
            raise NotImplemented
            
        new = copy.deepcopy(self)
        for comp in new.components:
            newcomp = getattr(self,comp) + getattr(other,comp)
            new._set_component(comp,newcomp)
        
        new._setupInterpolator()
        return new
    
    def __sub__(self, other):
        if not isinstance(other,Field):
            raise NotImplemented
            
        new = copy.deepcopy(self)
        for comp in new.components:
            newcomp = getattr(self,comp) - getattr(other,comp)
            new._set_component(comp,newcomp)
        
        new._setupInterpolator()
        return new
    
    def __rmul__(self, other):
        if not (isinstance(other,float) or isinstance(other,int)):
            raise NotImplemented
            
        new = copy.deepcopy(self)
        for comp in new.components:
            newcomp = getattr(self,comp) * other
            new._set_component(comp,newcomp)
        
        new._setupInterpolator()
        return new
            
    def __mul__(self, other):
        return self.__rmul__(other)