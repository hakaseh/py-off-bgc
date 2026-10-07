import taichi as ti
import numpy as np

# Mapping: Model Standard Name -> GLODAP File Name
GLODAP_MAP = {
    'nitrate': 'NO3',
    'phosphate': 'PO4',
    'silicate': 'silicate',
    'oxygen': 'oxygen',
    'dic': 'TCO2',
    'alk': 'TAlk'
}

@ti.kernel
def calculate_par_ti(
    sw_surface: ti.types.ndarray(), 
    psum: ti.types.ndarray(), 
    dz: ti.types.ndarray(), 
    par: ti.types.ndarray()
):
    """
    Beer-Lambert Law for light attenuation.
    Calculates optical depth column-by-column, perfectly parallelized across the 2D surface.
    """
    nz, ny, nx = psum.shape
    
    for j, i in ti.ndrange(ny, nx):
        if dz[0, j, i] > 1e-6:
            # 1. Surface light (approx 43% of SW radiation)
            light = sw_surface[j, i] * 0.43
            
            # 2. Iterate down the water column
            for k in range(nz):
                if dz[k, j, i] < 1e-6:
                    break  # Hit the seafloor, stop calculating light
                    
                # k_total = k_water + (k_phyto * phytoplankton_sum)
                k_total = 0.04 + 0.03 * psum[k, j, i]
                tau = k_total * dz[k, j, i]
                
                # PAR at the center of the current cell
                par[k, j, i] = light * ti.math.exp(-0.5 * tau)
                
                # Attenuate the light for the TOP of the next cell down
                light = light * ti.math.exp(-tau)


@ti.kernel
def apply_sinking_ti(
    tracer: ti.types.ndarray(), 
    dz: ti.types.ndarray(), 
    dt: ti.f32, 
    w_sink: ti.f32, 
    min_dz: ti.f32
):
    """
    1D Upwind Vertical Sinking with Adaptive Sub-stepping.
    Runs 100% in-place memory by passing flux down the column iteratively.
    """
    # Convert speed: m/day -> m/s
    w_s = w_sink / 86400.0
    
    if w_s > 0.0:
        # Calculate safe sub-step (Max distance per step = 90% of thinnest cell)
        dt_safe = (min_dz / ti.max(w_s, 1e-12)) * 0.9 
        n_steps = ti.max(1, ti.cast(ti.ceil(dt / dt_safe), ti.i32))
        dt_sub = dt / float(n_steps)
        
        nz, ny, nx = tracer.shape
        
        # Parallelize across the 2D surface
        for j, i in ti.ndrange(ny, nx):
            if dz[0, j, i] > 1e-6:
                
                # Execute the CFL-safe sub-steps
                for step in range(n_steps):
                    
                    # Surface flux is always 0 (no atmospheric influx of organic matter)
                    flux_in = 0.0
                    
                    # Iterate top-to-bottom
                    for k in range(nz):
                        if dz[k, j, i] < 1e-6:
                            break
                        
                        # Calculate flux leaving the bottom of this cell
                        flux_out = w_s * tracer[k, j, i]
                        
                        # Change in concentration
                        trend = (flux_in - flux_out) / dz[k, j, i]
                        
                        # Update the cell in-place (safely clamped to 0)
                        tracer[k, j, i] = ti.max(0.0, tracer[k, j, i] + trend * dt_sub)
                        
                        # The flux leaving this cell becomes the flux entering the next cell down!
                        flux_in = flux_out