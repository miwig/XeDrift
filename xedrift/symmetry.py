import numpy as np

#this file contains code to expand a reduced number of parameters used for fitting with symmetries
#to the full n_phi * n_z parameters needed for plotting/calculating superpositions

def n_fold_symmetry(n,coeffs):
    """Concatenate n copies of the list of coeffs to get n-fold symmetry"""
    return n*coeffs

def sliding_fixed_foreach_z(n_phi,coeffs):
    """Coefficients for alternating fixed and sliding reflectors: (sz0,fz0,sz1,fz0,...)->(s,f) * n_phi/2"""
    q_sliding = coeffs[0::2]
    q_fixed = coeffs[1::2]
    qs = np.concatenate((q_sliding, q_fixed)).tolist()
    return n_fold_symmetry(int(n_phi/2),qs)

def single_fixed_n_sliding(n_phi,n_z,n,coeffs):
    """Given n_z coefficients for a fixed reflector (f) and n*n_z coefficients for sliding reflectors (s1,...,sn),
    alternate between sliding and (always the same) fixed, then repeat the pattern to fill the cylinder, giving n_phi/(2*n)-fold symmetry:
    (f,s1,s2,...,sn)->(s1,f,s2,f,...,sn,f) * n_phi/(2*n)"""
    qs_fixed = coeffs[:n_z]
    qs_sliding = (coeffs[n_z*i:n_z*(i+1)] for i in range(1,n+1))

    qs = ((q_sliding , qs_fixed) for q_sliding in qs_sliding)
    qs = np.concatenate(list(qs)).flatten().tolist()
    qs = n_fold_symmetry(int(n_phi/(2*n)),qs)
    assert(len(qs) == n_phi*n_z)
    return qs

def n_sliding_only(n_phi,n_z,n,coeffs):
    """As `single_fixed_n_sliding`, but set coefficients for fixed reflectors to zero:
    (s1,s2,...,sn)->(s1,0,s2,0,...,sn,0) * n_phi/(2*n)
    """
    return single_fixed_n_sliding(n_phi,n_z,n,[0]* n_z + coeffs)