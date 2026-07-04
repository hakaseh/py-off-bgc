import jax
import jax.numpy as jnp

# Mapping: Model Standard Name -> GLODAP File Name
GLODAP_MAP = {
    'nitrate': 'NO3',
    'phosphate': 'PO4',
    'silicate': 'silicate',
    'oxygen': 'oxygen',
    'dic': 'TCO2',
    'alk': 'TAlk'
}


@jax.jit
def calculate_par(sw_surface, psum, dz):
    """
    Beer-Lambert Law for light attenuation.
    Shared by all models.
    Fully vectorized using cumulative optical depth.
    """
    k_water = 0.04
    k_phyto = 0.03
    
    # 1. Surface light (approx 43% of SW radiation)
    # Expand 2D surface to 3D for matrix broadcasting: shape (1, ny, nx)
    light_surface = (sw_surface * 0.43)[None, :, :]
    
    # 2. Calculate attenuation coefficient for every cell simultaneously
    k_total = k_water + k_phyto * psum
    
    # 3. Calculate "Optical Depth" (tau) of each layer
    tau = k_total * dz
    
    # 4. Cumulative optical depth down the column
    # jnp.cumsum instantly sums down the z-axis (axis=0)
    cum_tau_bottom = jnp.cumsum(tau, axis=0)
    
    # The optical depth at the TOP of the layer is the bottom minus this layer's tau
    cum_tau_top = cum_tau_bottom - tau
    
    # 5. Calculate PAR at the center of each layer
    # Light at top of layer * exponential decay through half the layer
    light_at_top = light_surface * jnp.exp(-cum_tau_top)
    par = light_at_top * jnp.exp(-0.5 * tau)
    
    # 6. Mask out land cells
    return jnp.where(dz > 1e-6, par, 0.0)


@jax.jit
def apply_sinking(tracer, dz, dt, w_sink):
    """
    Generic Vertical Sinking (Upwind Scheme) with Adaptive Sub-stepping.
    Equation: dC/dt = -d(w*C)/dz
    Vectorized and compiled using jax.lax.fori_loop.
    """
    # Convert speed: m/day -> m/s
    w_s = w_sink / 86400.0
    
    # 1. Find minimum dz for CFL safety limit
    # Force land cells (dz <= 1e-6) to a huge number so they don't trigger the min()
    dz_safe_min = jnp.where(dz > 1e-6, dz, 1e6)
    min_dz = jnp.min(dz_safe_min)
    
    # 2. Calculate safe sub-step (Max distance per step = 90% of thinnest cell)
    # We use a tiny denominator clamp to prevent division by zero if w_s is exactly 0.0
    dt_safe = (min_dz / jnp.maximum(w_s, 1e-12)) * 0.9 
    
    # Calculate integer number of steps (minimum of 1)
    n_steps = jnp.maximum(1, jnp.ceil(dt / dt_safe).astype(jnp.int32))
    dt_sub = dt / n_steps
    
    # Safe divisor for the concentration update
    dz_safe_div = jnp.where(dz > 1e-6, dz, 1.0)
    
    # 3. Define the single sub-step logic for JAX's loop
    def step_fn(i, current_tracer):
        # Pad surface with zero (no atmospheric influx of particulate matter)
        zero_surface = jnp.zeros_like(current_tracer[0:1, :, :])
        tracer_padded = jnp.concatenate([zero_surface, current_tracer], axis=0)
        
        # Instant flux calculations for all vertical interfaces
        flux_in_top = w_s * tracer_padded[:-1, :, :]
        flux_out_bottom = w_s * tracer_padded[1:, :, :]
        
        # Apply trend
        trend = (flux_in_top - flux_out_bottom) / dz_safe_div
        updated_tracer = current_tracer + (trend * dt_sub)
        
        # Clamp to prevent negative concentrations
        return jnp.maximum(0.0, updated_tracer)
        
    # 4. Execute the XLA-compiled dynamic loop
    # lax.fori_loop is JAX's ultra-fast equivalent of `for i in range(n_steps):`
    tracer_final = jax.lax.fori_loop(0, n_steps, step_fn, tracer)
    
    # 5. Mask land and apply a bypass if sinking speed is 0.0
    tracer_masked = jnp.where(dz > 1e-6, tracer_final, tracer)
    return jnp.where(w_s > 0.0, tracer_masked, tracer)