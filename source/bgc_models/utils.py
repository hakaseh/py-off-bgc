import numpy as np
import math
from numba import njit, prange

# Mapping: Model Standard Name -> GLODAP File Name
GLODAP_MAP = {
    'nitrate': 'NO3',
    'phosphate': 'PO4',
    'silicate': 'silicate',
    'oxygen': 'oxygen',
    'dic': 'TCO2',
    'alk': 'TAlk'
}


@njit(parallel=True, fastmath=True)
def calculate_par(sw_surface, psum, dz):
    """
    Beer-Lambert Law for light attenuation.
    Shared by all models.
    Uses a hard-coded light attenuation coefficients of 0.04 m-1 for water and 0.03 m2 mg-1 for chlorophyll
    """
    nz, ny, nx = dz.shape
    
    # Use zeros_like to inherit shape and dtype, ensuring land cells default to 0.0
    par = np.zeros_like(dz)
    
    # Coefficients
    k_water = 0.04
    k_phyto = 0.03

    n_points = ny * nx

    for p in prange(n_points):
        j = p // nx
        i = p % nx
        
        # Surface PAR (approx 43% of SW radiation)
        light = sw_surface[j, i] * 0.43
        
        for k in range(nz):
            layer_depth = dz[k, j, i]
            
            # Skip masked cells / land
            if layer_depth <= 1e-6:
                continue
                
            # Attenuation
            k_total = k_water + k_phyto * psum[k, j, i]
            
            # Light at center of layer
            par[k, j, i] = light * np.exp(-0.5 * k_total * layer_depth)
            
            # Light entering next layer
            light = light * np.exp(-k_total * layer_depth)
            
    return par


@njit(parallel=True, fastmath=True)
def apply_sinking(tracer, dz, dt, w_sink):
    """
    Generic Vertical Sinking (Upwind Scheme) with Adaptive Sub-stepping.
    Equation: dC/dt = -d(w*C)/dz
    """
    nz, ny, nx = tracer.shape
    
    # 1. Use .copy() instead of zeros_like so land/skipped cells aren't erased
    tracer_new = tracer.copy()
    
    # Convert speed: m/day -> m/s
    w_s = w_sink / 86400.0
    
    if w_s <= 0.0:
        return tracer_new  # Skip math if it doesn't sink
        
    # 2. Find minimum dz to calculate the CFL safety limit
    min_dz = 1e6
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                if 1e-6 < dz[k, j, i] < min_dz:
                    min_dz = dz[k, j, i]
                    
    # 3. Calculate safe sub-step (Max distance per step = 90% of thinnest cell)
    dt_safe = (min_dz / w_s) * 0.9 
    n_steps = int(np.ceil(dt / dt_safe)) # Use np.ceil for pure NumPy dependency
    dt_sub = dt / n_steps  # The actual safe time step we will use
    
    n_points = ny * nx
    
    # Pre-allocate buffer OUTSIDE the loop to prevent memory thrashing
    tracer_current = np.empty_like(tracer)
    
    # 4. Perform Sinking in Safe, Bite-Sized Steps
    for step in range(n_steps):
        # Update buffer values in-place instead of re-allocating memory
        tracer_current[:] = tracer_new[:]
        
        for p in prange(n_points):
            j = p // nx
            i = p % nx
            
            # Top Layer (k=0): No flux from above, loses mass to below
            if dz[0, j, i] > 1e-6:
                loss = (w_s * tracer_current[0, j, i]) / dz[0, j, i]
                tracer_new[0, j, i] = tracer_current[0, j, i] - (loss * dt_sub)
            
            # Interior Layers
            for k in range(1, nz):
                if dz[k, j, i] > 1e-6:
                    # Gain from above (k-1), Lose to below (k)
                    gain = (w_s * tracer_current[k-1, j, i]) / dz[k, j, i]
                    loss = (w_s * tracer_current[k, j, i])   / dz[k, j, i]
                    
                    tracer_new[k, j, i] = tracer_current[k, j, i] + dt_sub * (gain - loss)
                    
    return tracer_new