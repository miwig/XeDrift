import importlib
import pickle
import pandas as pd
import time, os
import numpy as np
import matplotlib.pyplot as plt

from xedrift.tpcClass import TPC_X1T
import xedrift
from xedrift.chargeMatching import *


import sys
# Parameters

print("Hello!")

model_dir = './models' #path to electric field data
data_dir = './data' #path to experimental data

n_iterations=1000 #number of matching iterations to run

tpc_x1t = TPC_X1T() #for drift velocity, tpc walls for out of bounds condition
model_name = 'XE1T_3D_sp_eff_pot_48x6' #folder with electric field data inside model_dir

charges_n_phi = 48 #number of angular subdivisions for charge basis vectors
charges_n_z = 6 #number of vertical subdivisions for charge basis vectors

nudge_single = True #only change one coefficient per iteration (gibbs sampling)
nudge_scale = 0.2 #gaussian proposal distribution sigma

data_name = 'kr83m/kr83m_sr1_{}'.format(sys.argv[1]) #folder with experimental data in data_dir, get dataset id from command line argument

num_coefficients = charges_n_z * (charges_n_phi/2)/4 #number of coefficients for matching, /2 for uncharged fixed reflectors, /4 for 90 degree symmetry 

state_0 = np.array([0.0] * num_coefficients) #initial state: zero charge


import itertools

#create rectangular grid in observation space for tetrahedralization
xys = np.linspace(-46,46,12)
dts = np.linspace(0,780,10)
grid = np.array(list(itertools.product(xys,xys,dts)))

# End parameters

model_path = model_dir + model_name
data_path = data_dir + data_name

results_dir = '{}/matchresults/{}/{}/'.format(data_dir,data_name,model_name) #save results here

make_dir(results_dir)

print("Results in: " + results_dir)

# Prepare experimental data

data = pd.read_hdf(data_path + '.hdf')
data['drift_time_us'] = data.drift_time/1e3

grid = preprocess_grid(grid, data) #Remove grid points that fall outside the data
triangulation = triangulate_grid(grid)

simp_counts = count_events(triangulation,data) #count events in observation space simplices
tri_mask = mask_triangulation(triangulation,grid,simp_counts) #sanity checks on simplex aspect ratio, volume and event count
simps_obs = triangulation.simplices.copy()[tri_mask]

print("Events in triangulation: {:.2f}%".format(simp_counts.events.sum()/len(data)))
print("Events in masked triangulation: {:.2f}%".format(simp_counts[tri_mask].events.sum()/len(data)))

#superpos = makeSuperposition(model_path,charges_n_phi,charges_n_z) #code for matching without symmetry

#load electric field data (basis vectors) with uncharged fixed reflectors, 90 degree symmetry
fields = [model_path + '/no_charge/electric_field.txt']
for i in range(num_coefficients):
    fields += [model_path + '/symmetry_90_no_fixed/field_{}.npz'.format(i)]
    
assert(len(state_0)==len(fields)-1)
    
superpos = Superposition(fields) #load electric field data (basis vectors)


from functools import partial

log_l_f = partial(log_like_no_pool,tpc_x1t,superpos,grid,triangulation,tri_mask,simp_counts) #log-likelihood function for metropolis algorithm

match_data = init_match_data(results_dir, log_l_f, state_0) #set initial state or load saved state from previous run if found


state = match_data.states[-1]
log_l = match_data.log_ls[-1]

for i in range(n_iterations):
    start_time = time.time()
    
    state, log_l, accepted, *meta = MHStep(state,log_l_f,log_l,scale=nudge_scale,nudgeSingle=nudge_single) #single step of metropolis algorithm
    if(accepted):
        print("Accepted! {}".format("single" if nudge_single else "all"))
        
    time_elapsed = (time.time() - start_time)
    
    match_data.states.append(state)
    match_data.log_ls.append(log_l)
    match_data.accs.append(accepted)
    match_data.metas.append(meta)
    print("---------------- {} | {:.2f} | {:.2f}%".format(i,time_elapsed,100*sum(match_data.accs[-50:])/50))
    
    if(accepted): #save results
        pickle.dump({'match_data' : match_data, 'model_name' : model_name, 'data_name' : data_name, 'grid' : grid, 'tri_data' : (triangulation.simplices,tri_mask), 'simp_counts': simp_counts, 'symmetry' : 'sym486_90'} ,open(results_dir+'/matchresult.p','wb'))
