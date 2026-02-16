import numpy as np
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
    """
    nz, ny, nx = dz.shape
    par = np.zeros((nz, ny, nx))
    
    # Coefficients (Make sure these match your physics!)
    k_water = 0.04
    k_phyto = 0.03

    for j in prange(ny):
        for i in range(nx):
            # Surface PAR (approx 43% of SW radiation)
            light = sw_surface[j, i] * 0.43
            
            for k in range(nz):
                # Attenuation
                k_total = k_water + k_phyto * psum[k, j, i]
                layer_depth = dz[k, j, i]
                
                # Light at center of layer
                par[k, j, i] = light * np.exp(-0.5 * k_total * layer_depth)
                
                # Light entering next layer
                light = light * np.exp(-k_total * layer_depth)
                
    return par

@njit(parallel=True, fastmath=True)
def apply_sinking(tracer, dz, dt, w_sink):
    """
    Generic Vertical Sinking (Upwind Scheme).
    Equation: dC/dt = -d(w*C)/dz
    
    Args:
        tracer: 3D array (nz, ny, nx)
        dz: 3D array (nz, ny, nx) - Vertical grid spacing
        dt: Time step (seconds)
        w_sink: Sinking speed (m/day) -> MUST BE CONVERTED TO m/s inside!
    """
    nz, ny, nx = tracer.shape
    tracer_new = np.zeros_like(tracer)
    
    # Convert speed: m/day -> m/s
    w_s = w_sink / 86400.0
    
    for j in prange(ny):
        for i in range(nx):
            # Top Layer (k=0): No flux from above, loses mass to below
            if dz[0, j, i] > 1e-6:
                loss = (w_s * tracer[0, j, i]) / dz[0, j, i]
                tracer_new[0, j, i] = tracer[0, j, i] - (loss * dt)
            
            # Interior Layers
            for k in range(1, nz):
                if dz[k, j, i] > 1e-6:
                    # Gain from above (k-1), Lose to below (k)
                    # Flux In  = w * C_above
                    # Flux Out = w * C_current
                    gain = (w_s * tracer[k-1, j, i]) / dz[k, j, i]
                    loss = (w_s * tracer[k, j, i])   / dz[k, j, i]
                    
                    tracer_new[k, j, i] = tracer[k, j, i] + dt * (gain - loss)
                    
    return tracer_new