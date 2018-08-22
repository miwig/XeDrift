import numpy as np
import re
from scipy.interpolate import RegularGridInterpolator
import copy

class Field:
    """
    Represents a multi-dimensional field given on a grid

    Allows interpolation of field values for off-grid positions,
    calculation of superpositions of multiple fields and reading COMSOL exported .txt
    """

    def _set_component_index(self,name,index):
        self.components[name] = index

    def _get_component_slice(self,index):
        return (slice(None),)*self.dim_space + (index,) #slice to take everything along dim_space spatial axes, but only component with index index

    def _set_components_convenience(self):
        for comp, idx in self.components.items():
            setattr(self,comp,self.field_values[self._get_component_slice(idx)])

    def _set_grid_convenience(self):
        if(self.dim_space == 2):
            self.rs, self.zs = self.grid[0], self.grid[1]
        elif(self.dim_space==3):
            self.xs, self.ys, self.zs = self.grid[0], self.grid[1], self.grid[2]

    def _from_comsol_file(self,filename):
        """Build instance from a COMSOL .txt grid export file"""

        file = open(filename,'r')

        while True:
            line = file.readline()
            if not line:
                break

            if '% Dimension' in line:
                self.dim_space = int(line[-2]) #-1 is \n, -2 is dimension

            if '% Expressions' in line:
                self.dim_field = int(line[-2]) #-1 is \n, -2 is dimension

            if not self.grid and '% Grid' in line:
                self.grid = tuple(np.genfromtxt(file,max_rows=1)/10 for i in range(self.dim_space))#convert mm to cm
                self.shape = tuple(len(dim) for dim in self.grid)
                self.field_values = np.empty(shape=self.shape+(self.dim_field,))

                continue

            if '% Data' in line:
                self._read_comsol_data(file)
                continue

    def _read_comsol_data(self,file):
        """Read actual field data from a COMSOL .txt grid export file"""

        attr = file.readline()
        reg = r'\.(.+?)\s'
        attr = re.search(reg,attr).group(1)
        
        numrows = np.prod(self.shape[1:]) #for 2d, read as many rows as there are z values, for 3d, read #y*#z rows
        #np.genfromtxt is considerate and reads only the lines it needs to, leaving the file handle intact
        data = np.transpose(np.genfromtxt(file,max_rows=numrows))
        if(self.dim_space > 2):
            #‘F’ means to read / write the elements using Fortran-like index order, with the first index changing fastest, and the last index changing slowest.
            #seems to match comsol behaviour
            data = np.reshape(data, self.shape,order='F')
        
        comp_idx = len(self.components)
        self._set_component_index(attr,comp_idx)
        comp_slice = self._get_component_slice(comp_idx)
        self.field_values[comp_slice] = data

    def _from_grid_values(self,grid,values,components):
        """Build instance from grid and field values as numpy arrays"""

        self.grid = grid
        self.field_values = values
        self.dim_space = grid.shape[0]
        self.dim_field = values.shape[-1]
        self.shape = tuple(len(dim) for dim in self.grid)

        for comp_idx, comp in enumerate(components):
            self._set_component_index(comp,comp_idx)

    def __init__(self,filename=None,gridspec=None):
        self.components = {}
        self.interpolator = {}
        self.dim_space = -1
        self.dim_field = -1
        self.grid = None
        
        if filename is None:
            self._from_grid_values(gridspec[0],gridspec[1],gridspec[2])
        else:
            self._from_comsol_file(filename)

        self._finalize()

    def _setup_interpolator(self):
        self.interpolator = RegularGridInterpolator(self.grid,self.field_values)

    def _finalize(self):
        self._setup_interpolator()
        self._set_components_convenience()
        self._set_grid_convenience()
             
    def getValue(self,pos):
        try:
            return self.interpolator(pos)
        except ValueError:
            print("Error at {}".format(pos))
            return np.zeros(self.dim_field)

    def _check_operator_valid(self,other):
        if not isinstance(other,Field):
            raise NotImplemented

        if not np.shape(other.field_values) == np.shape(self.field_values):
            raise ValueError

    def __add__(self, other):
        self._check_operator_valid(other)
            
        new = copy.deepcopy(self)
        new.field_values = self.field_values + other.field_values
        
        new._finalize()
        return new
    
    def __sub__(self, other):
        self._check_operator_valid(other)
            
        new = copy.deepcopy(self)
        new.field_values = self.field_values - other.field_values
        
        new._finalize()
        return new
    
    def __rmul__(self, other):
        if not (isinstance(other,float) or isinstance(other,int)):
            raise NotImplemented
            
        new = copy.deepcopy(self)
        new.field_values = self.field_values * other
        
        new._finalize()
        return new
            
    def __mul__(self, other):
        return self.__rmul__(other)