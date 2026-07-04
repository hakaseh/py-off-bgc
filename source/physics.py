import numpy as np
from numba import njit, prange
import math
import jax
import jax.numpy as jnp

@jax.jit(static_argnums=(8,))
def advection_neumann_jax(tracer, u, v, w, dz, dt, dx, dy, is_global=False):
    # Expand 2D metrics to 3D for matrix broadcasting
    dx_c = dx[None, :, :]
    dy_c = dy[None, :, :]

    # --- 1. Horizontal Neighbors (Array Rolling) ---
    # Shift the whole 3D array to simulate looking East, West, North, and South
    tr_west = jnp.roll(tracer, shift=1, axis=2)
    tr_east = jnp.roll(tracer, shift=-1, axis=2)
    dz_west = jnp.roll(dz, shift=1, axis=2)
    dz_east = jnp.roll(dz, shift=-1, axis=2)

    tr_south = jnp.roll(tracer, shift=1, axis=1)
    tr_north = jnp.roll(tracer, shift=-1, axis=1)
    dz_south = jnp.roll(dz, shift=1, axis=1)
    dz_north = jnp.roll(dz, shift=-1, axis=1)

    # Neumann boundaries: if neighbor is land (dz < 1e-6), use own center value
    c_west = jnp.where(dz_west > 1e-6, tr_west, tracer)
    c_east = jnp.where(dz_east > 1e-6, tr_east, tracer)
    c_south = jnp.where(dz_south > 1e-6, tr_south, tracer)
    c_north = jnp.where(dz_north > 1e-6, tr_north, tracer)

    # Upwind logic evaluated for the entire grid simultaneously
    c_up_x = jnp.where(u > 0.0, c_west, c_east)
    c_up_y = jnp.where(v > 0.0, c_south, c_north)

    term_x = jnp.abs(u) * (tracer - c_up_x) / dx_c
    term_y = jnp.abs(v) * (tracer - c_up_y) / dy_c

    # --- 2. Vertical Neighbors (Array Slicing) ---
    # Pad the top and bottom to safely shift the z-axis
    tr_above = jnp.concatenate([tracer[0:1, :, :], tracer[:-1, :, :]], axis=0)
    tr_below = jnp.concatenate([tracer[1:, :, :], tracer[-1:, :, :]], axis=0)
    dz_above = jnp.concatenate([dz[0:1, :, :], dz[:-1, :, :]], axis=0)
    dz_below = jnp.concatenate([dz[1:, :, :], dz[-1:, :, :]], axis=0)

    # Center vertical velocity from faces
    w_center = 0.5 * (w[:-1, :, :] + w[1:, :, :])

    c_above = jnp.where(dz_above > 1e-6, tr_above, tracer)
    c_below = jnp.where(dz_below > 1e-6, tr_below, tracer)

    term_z = jnp.where(w_center > 0.0,
                       w_center * (tracer - c_below) / dz,
                       w_center * (c_above - tracer) / dz)

    # --- 3. Total Change ---
    raw_c = tracer - dt * (term_x + term_y + term_z)
    tracer_new = jnp.maximum(0.0, raw_c)

    # Mask out land domains
    return jnp.where(dz > 1e-6, tracer_new, tracer)

@jax.jit
def calculate_full_kz_jax(mld_2d, rho_3d, dz_3d, k_mld_max=1e-2, k_bg_min=1e-6, k_deep_max=1e-5, k_boundary=1e-4):
    # Instantly calculate depth for the whole 3D grid
    current_depth = jnp.cumsum(dz_3d, axis=0)
    mld_3d = mld_2d[None, :, :]

    # --- 1. Dynamic Boundary Detection ---
    dz_below = jnp.concatenate([dz_3d[1:, :, :], jnp.zeros_like(dz_3d[-1:, :, :])], axis=0)
    is_bottom = dz_below <= 1e-6

    dz_west = jnp.roll(dz_3d, shift=1, axis=2)
    dz_east = jnp.roll(dz_3d, shift=-1, axis=2)
    dz_south = jnp.roll(dz_3d, shift=1, axis=1)
    dz_north = jnp.roll(dz_3d, shift=-1, axis=1)
    
    is_coast = (dz_west <= 1e-6) | (dz_east <= 1e-6) | (dz_south <= 1e-6) | (dz_north <= 1e-6)
    is_boundary = is_bottom | is_coast

    local_bg_min = jnp.where(is_boundary, k_boundary, k_bg_min)
    local_ceiling = jnp.where(is_boundary, k_boundary, k_deep_max)

    # --- 2. Mixing Physics ---
    # Inside MLD
    sigma = current_depth / mld_3d
    shape = sigma * (1.0 - sigma)**2
    kz_mld = local_bg_min + (k_mld_max * 6.75 * shape)

    # Below MLD (Stratification)
    rho_below = jnp.concatenate([rho_3d[1:, :, :], rho_3d[-1:, :, :]], axis=0)
    drho = rho_below - rho_3d
    dz_eff = 0.5 * (dz_3d + dz_below)
    
    # Calculate N2, defaulting bottom-most cells to 1e-7
    N2 = jnp.maximum((9.81 / 1035.0) * (drho / dz_eff), 1e-7)
    N2 = N2.at[-1, :, :].set(1e-7)

    k_deep = 1e-11 / jnp.sqrt(N2)
    kz_deep = jnp.clip(k_deep, local_bg_min, local_ceiling)

    # --- 3. Combine and Mask ---
    kz_3d = jnp.where(current_depth <= mld_3d, kz_mld, kz_deep)
    return jnp.where(dz_3d > 1e-6, kz_3d, jnp.nan)

@jax.jit
def diffusion_robust_jax(tracer, k_z, dz_3d, dt):
    epsilon = 1e-6

    # Calculate interfaces between layers (k and k+1)
    delta_z = 0.5 * (dz_3d[:-1, :, :] + dz_3d[1:, :, :])
    k_face = 0.5 * (k_z[:-1, :, :] + k_z[1:, :, :])

    # Safety Stability Clamp
    k_max = 0.45 * (delta_z ** 2) / dt
    k_eff = jnp.minimum(k_face, k_max)

    # Flux calculation for all inner interfaces
    grad = (tracer[1:, :, :] - tracer[:-1, :, :]) / delta_z
    flux_inner = jnp.where(delta_z >= epsilon, k_eff * grad, 0.0)

    # Pad top and bottom interfaces with 0 flux (No atmospheric/seafloor escape)
    zero_flux = jnp.zeros_like(flux_inner[0:1, :, :])
    flux = jnp.concatenate([zero_flux, flux_inner, zero_flux], axis=0)

    # Divergence: Flux In (bottom) - Flux Out (top)
    trend = (flux[1:, :, :] - flux[:-1, :, :]) / dz_3d

    # Apply trend to volume cells, ignore land
    tracer_out = jnp.where(dz_3d >= epsilon, tracer + dt * trend, tracer)
    return tracer_out

@jax.jit
def haversine_dist(lon1, lat1, lon2, lat2):
    """
    Calculates distance (meters) between two points on Earth.
    Inputs in DEGREES. Can handle both single numbers and full arrays!
    """
    R = 6371000.0 # Earth Radius (m)
    
    # Convert to Radians using JAX numpy
    phi1 = jnp.radians(lat1)
    phi2 = jnp.radians(lat2)
    dphi = jnp.radians(lat2 - lat1)
    dlam = jnp.radians(lon2 - lon1)
    
    a = jnp.sin(dphi/2.0)**2 + jnp.cos(phi1) * jnp.cos(phi2) * jnp.sin(dlam/2.0)**2
    c = 2.0 * jnp.arctan2(jnp.sqrt(a), jnp.sqrt(1.0 - a))
    
    return R * c

@jax.jit
def calculate_metrics_curvilinear(lon_2d, lat_2d):
    """
    Calculates dx and dy (meters) for a 2D curvilinear grid.
    dx[j,i] is the distance to the EAST neighbor (i+1).
    dy[j,i] is the distance to the NORTH neighbor (j+1).
    """
    
    # --- DX (Distance along I-axis: East-West) ---
    # Calculate distances between (i) and (i+1) for all columns except the very last one
    dx_inner = haversine_dist_jax(
        lon_2d[:, :-1], lat_2d[:, :-1],  # Base points
        lon_2d[:, 1:],  lat_2d[:, 1:]    # East neighbors
    )
    
    # For the rightmost boundary, duplicate the previous distance (Numba's 'else' condition)
    # dx_inner[:, -1:] selects the last calculated column while keeping its 2D shape intact
    dx_last = dx_inner[:, -1:]
    
    # Stitch the inner distances and the boundary boundary column together
    dx = jnp.concatenate([dx_inner, dx_last], axis=1)

    # --- DY (Distance along J-axis: North-South) ---
    # Calculate distances between (j) and (j+1) for all rows except the very top one
    dy_inner = haversine_dist_jax(
        lon_2d[:-1, :], lat_2d[:-1, :],  # Base points
        lon_2d[1:, :],  lat_2d[1:, :]    # North neighbors
    )
    
    # For the topmost boundary, duplicate the previous distance
    dy_last = dy_inner[-1:, :]
    
    # Stitch the inner distances and the boundary row together
    dy = jnp.concatenate([dy_inner, dy_last], axis=0)

    # --- Safety Clamp ---
    # Instantly clamp all values below 1.0 across the entire grid
    dx = jnp.maximum(dx, 1.0)
    dy = jnp.maximum(dy, 1.0)

    return dx, dy
    
@jax.jit
def calculate_dz_from_centers(depths_1d):
    """
    Calculates cell thicknesses from center depths.
    Fully vectorized, zero for loops.
    """
    # 1. Calculate all inner interfaces simultaneously
    midpoints = 0.5 * (depths_1d[:-1] + depths_1d[1:])
    
    # 2. Define the surface (0.0) and calculate the absolute bottom
    surface = jnp.array([0.0])
    bottom = jnp.array([depths_1d[-1] + (depths_1d[-1] - midpoints[-1])])
    
    # 3. Stitch them all together (Surface + Midpoints + Bottom)
    interfaces = jnp.concatenate([surface, midpoints, bottom])
    
    # 4. Instant diff across the entire array
    return jnp.diff(interfaces)

@jax.jit
def _calculate_grid_metrics_kernel(lon_2d, lat_2d):
    """
    The pure-math JAX core. 
    Requires pre-formatted 2D arrays.
    """
    R_EARTH = 6371000.0
    
    # JAX has the exact same radians and gradient functions as NumPy
    lat_rad = jnp.radians(lat_2d)
    lon_rad = jnp.radians(lon_2d)
    
    dlat = jnp.gradient(lat_rad, axis=0)
    dy = R_EARTH * dlat
    
    dlon = jnp.gradient(lon_rad, axis=1)
    dx = R_EARTH * jnp.cos(lat_rad) * dlon
    
    return jnp.abs(dx), jnp.abs(dy)


def calculate_grid_metrics(ds):
    """
    The Python Wrapper. 
    Handles xarray logic, then passes pure arrays to JAX.
    """
    lat = ds['lat'].values
    lon = ds['lon'].values
    
    # Handle the shape logic in standard Python
    if lat.ndim == 1: 
        lon_2d, lat_2d = jnp.meshgrid(lon, lat)
    else: 
        lon_2d, lat_2d = lon, lat
        
    # Call the compiled XLA math
    return _calculate_grid_metrics_kernel(lon_2d, lat_2d)

@jax.jit
def calculate_w_rigid_lid(u, v, dz_3d, dx, dy):
    nz, ny, nx = u.shape

    # Expand 2D grid spacing to 3D for matrix broadcasting
    dx_c = dx[None, :, :]
    dy_c = dy[None, :, :]

    # --- 1. HORIZONTAL DIVERGENCE ---
    # Shift arrays to get raw face velocities
    u_west_raw  = 0.5 * (jnp.roll(u, shift=1, axis=2) + u)
    u_east_raw  = 0.5 * (u + jnp.roll(u, shift=-1, axis=2))
    v_south_raw = 0.5 * (jnp.roll(v, shift=1, axis=1) + v)
    v_north_raw = 0.5 * (v + jnp.roll(v, shift=-1, axis=1))

    # Mask face velocities if a neighbor is land (dz <= 1e-6)
    dz_west  = jnp.roll(dz_3d, shift=1, axis=2)
    dz_east  = jnp.roll(dz_3d, shift=-1, axis=2)
    dz_south = jnp.roll(dz_3d, shift=1, axis=1)
    dz_north = jnp.roll(dz_3d, shift=-1, axis=1)

    u_west  = jnp.where(dz_west > 1e-6, u_west_raw, 0.0)
    u_east  = jnp.where(dz_east > 1e-6, u_east_raw, 0.0)
    v_south = jnp.where(dz_south > 1e-6, v_south_raw, 0.0)
    v_north = jnp.where(dz_north > 1e-6, v_north_raw, 0.0)

    # Calculate Horizontal Divergence
    div_h = ((u_east - u_west) / dx_c) + ((v_north - v_south) / dy_c)

    # --- 2. INTEGRATE DOWNWARD ---
    # Change in W per layer (delta W). Zero out contributions from land cells.
    delta_w = jnp.where(dz_3d > 1e-6, div_h * dz_3d, 0.0)
    
    # Cumulative sum replaces the downward for-loop. 
    w_accum = jnp.cumsum(delta_w, axis=0)
    
    # Pad the surface layer with 0.0 to get the correct (nz+1) face interfaces
    w_surface = jnp.zeros((1, ny, nx), dtype=w_accum.dtype)
    w_raw = jnp.concatenate([w_surface, w_accum], axis=0)

    # --- 3. TRUE SEAFLOOR AND LINEAR CORRECTION ---
    # Counting the wet layers exactly equals the k_bottom index!
    k_bottom = jnp.sum(dz_3d > 1e-6, axis=0)

    # The error at the bottom is simply the total sum of delta_w down the column
    bottom_error = w_accum[-1, :, :]

    # Create a 3D vertical index grid (0 to nz)
    k_indices = jnp.arange(nz + 1)[:, None, None]

    # Calculate linear correction weights (k / k_bottom)
    # Prevent division-by-zero on pure land columns by defaulting to 0.0
    weight = jnp.where(k_bottom > 0, k_indices / k_bottom, 0.0)

    # Apply the linear correction simultaneously to all cells
    w_corrected = w_raw - (bottom_error[None, :, :] * weight)

    # --- 4. MASKING ---
    # 1. Force W to 0.0 strictly below the true seafloor
    valid_w_mask = k_indices <= k_bottom[None, :, :]
    w_final = jnp.where(valid_w_mask, w_corrected, 0.0)

    # 2. Force W to 0.0 on the outer lateral boundaries (replicating Numba's skip boundaries)
    mask_y = (jnp.arange(ny) > 0) & (jnp.arange(ny) < ny - 1)
    mask_x = (jnp.arange(nx) > 0) & (jnp.arange(nx) < nx - 1)
    inner_domain_mask = mask_y[:, None] & mask_x[None, :]

    return jnp.where(inner_domain_mask[None, :, :], w_final, 0.0)

def create_restoring_weights(water_mask, sponge_width, dt, tau_lateral=86400.0, tau_bottom=86400.0*30, tau_coast=86400.0, is_global=False):
    """
    Creates a 3D rate array (1/sec) for restoring.
    Combines Lateral Sponge (Fast), Seafloor Restoring (Slow), 
    and Coastal Restoring (for terrestrial nutrient runoff).
    """
    nz, ny, nx = water_mask.shape
    
    # 1. Initialize Rates with Zeros
    restore_rate = np.zeros((nz, ny, nx))
    
    # Define Rates (1/sec)
    rate_lateral = 1.0 / tau_lateral
    rate_bottom  = 1.0 / tau_bottom
    rate_coast   = 1.0 / tau_coast
    
    # --- A. LATERAL SPONGE (N/S/E/W) ---
    # Taper from edge (rate_lateral) to interior (0.0)
    for i in range(sponge_width):
        # Linear weight: 1.0 at edge, decreasing inwards
        weight = (sponge_width - i) / sponge_width
        val = weight * rate_lateral
        
        # Apply to North and South sides (Always active)
        restore_rate[:, i, :]      = np.maximum(restore_rate[:, i, :], val)       # South
        restore_rate[:, -(i+1), :] = np.maximum(restore_rate[:, -(i+1), :], val)  # North

        # Apply to East and West sides ONLY if regional (not global)
        if not is_global:
            restore_rate[:, :, i]      = np.maximum(restore_rate[:, :, i], val)       # West
            restore_rate[:, :, -(i+1)] = np.maximum(restore_rate[:, :, -(i+1)], val)  # East

    # --- B. SEAFLOOR RESTORING ---
    # We iterate over 2D surface to find the deepest wet cell k
    for j in prange(ny):
        for i in range(nx):
            col = water_mask[:, j, i]
            if not np.any(col):
                continue
                
            k_bot = np.where(col)[0][-1]
            restore_rate[k_bot, j, i] = np.maximum(restore_rate[k_bot, j, i], rate_bottom)

    # --- C. COASTAL RESTORING (Nutrient Runoff Proxy) ---
    # River runoff is buoyant and enters the ocean at the surface. 
    # Therefore, we only detect coastlines and apply restoring to the surface layer (k=0).
    
    coastal_mask_2d = np.zeros_like(water_mask[0], dtype=bool)
    surface_water = water_mask[0]
    
    # Matrix shift to detect land in all 4 compass directions
    coastal_mask_2d[1:, :] |= (~surface_water[:-1, :])  # Check South
    coastal_mask_2d[:-1, :] |= (~surface_water[1:, :])  # Check North
    coastal_mask_2d[:, 1:] |= (~surface_water[:, :-1])  # Check West
    coastal_mask_2d[:, :-1] |= (~surface_water[:, 1:])  # Check East
    
    # A coastal cell MUST actually be water
    coastal_mask_2d &= surface_water
    
    # Apply the coastal rate ONLY to the surface layer (k=0).
    # np.maximum ensures that if a coastal cell is ALSO inside the lateral sponge, the faster rate wins.
    restore_rate[0] = np.where(
        coastal_mask_2d, 
        np.maximum(restore_rate[0], rate_coast), 
        restore_rate[0]
    )

    # --- FINAL SCALING ---
    # Multiply by DT to get the "nudge fraction" per step
    # Result is a 3D array of alpha values: C_new = C_old + alpha * (C_target - C_old)
    nudge_coeff = restore_rate * dt
    
    # Stability Check: alpha cannot exceed 1.0 (100% replacement per timestep)
    nudge_coeff = np.minimum(nudge_coeff, 1.0)
    
    # Mask out land finally to ensure dry cells remain strictly 0.0
    nudge_coeff = np.where(water_mask, nudge_coeff, 0.0)
    
    return nudge_coeff

#Deprecated. To be deleted!
@njit(parallel=True, fastmath=True)
def mixing_convective(tracer, rho_3d, dz_3d, delta_rho_mld):
    """
    Vertical mixing based on density gradient (Convective Adjustment).
    Uses a 10m reference depth to bypass surface freshwater/heating lenses.
    """
    nz, ny, nx = tracer.shape
    tracer_out = np.empty_like(tracer)
    
    for j in prange(ny):
        for i in range(nx):
            # 1. LAND CHECK
            if dz_3d[0, j, i] < 1e-6 or np.isnan(tracer[0, j, i]):
                tracer_out[:, j, i] = np.nan
                continue
                
            # Copy column data to local 1D arrays
            col_tr = np.empty(nz)
            col_rho = np.empty(nz)
            for k in range(nz):
                col_tr[k] = tracer[k, j, i]
                col_rho[k] = rho_3d[k, j, i]
                
            # --- 1. MLD MIXING (10m Reference) ---
            # Find the reference layer closest to 10m
            depth_accum = 0.0
            k_ref = 0
            for k in range(nz):
                depth_accum += dz_3d[k, j, i]
                if depth_accum >= 10.0:
                    k_ref = k
                    break
                    
            rho_ref = col_rho[k_ref]
            
            # Find the bottom of the mixed layer
            k_mld = 0
            for k in range(nz):
                if k >= k_ref:
                    if (col_rho[k] - rho_ref) > delta_rho_mld:
                        k_mld = k - 1  # The layer just before the threshold was crossed
                        break
                    else:
                        k_mld = k  # Keep going down if still well-mixed
                        
            # Homogenize everything from the surface down to k_mld
            if k_mld > 0:
                sum_mass = 0.0
                sum_rho_mass = 0.0
                sum_vol = 0.0
                for k in range(k_mld + 1):
                    vol = dz_3d[k, j, i]
                    sum_mass += col_tr[k] * vol
                    sum_rho_mass += col_rho[k] * vol
                    sum_vol += vol
                
                avg_conc = sum_mass / sum_vol
                avg_rho = sum_rho_mass / sum_vol
                for k in range(k_mld + 1):
                    col_tr[k] = avg_conc
                    col_rho[k] = avg_rho
                    
            # --- 2. DEEP CONVECTIVE ADJUSTMENT (Instability) ---
            unstable = True
            passes = 0
            while unstable and passes < nz:
                unstable = False
                passes += 1
                for k in range(k_mld, nz - 1):
                    # Using potential density here is perfect!
                    if col_rho[k] > col_rho[k+1]:
                        vol1 = dz_3d[k, j, i]
                        vol2 = dz_3d[k+1, j, i]
                        total_vol = vol1 + vol2
                        
                        avg_tr = (col_tr[k]*vol1 + col_tr[k+1]*vol2) / total_vol
                        col_tr[k] = avg_tr
                        col_tr[k+1] = avg_tr
                        
                        avg_rho = (col_rho[k]*vol1 + col_rho[k+1]*vol2) / total_vol
                        col_rho[k] = avg_rho
                        col_rho[k+1] = avg_rho
                        
                        unstable = True
                        
            # Save stabilized column
            for k in range(nz):
                tracer_out[k, j, i] = col_tr[k]
                
    return tracer_out


@jax.jit
def calculate_mld(rho_3d, dz_3d, delta_rho_mld):
    nz, ny, nx = rho_3d.shape
    
    # --- 1. Calculate Depths ---
    # The depth at the BOTTOM of each cell
    bottom_depths_3d = jnp.cumsum(dz_3d, axis=0)
    
    # The depth at the TOP of each cell
    top_depths_3d = bottom_depths_3d - dz_3d
    
    # --- 2. Find the 10m Reference Layer ---
    # jnp.argmax on a boolean array returns the first index where the statement is True
    k_ref_2d = jnp.argmax(bottom_depths_3d >= 10.0, axis=0)
    
    # Extract the reference density for every column simultaneously
    # (We add [None, ...] to match 3D dimensions, then [0] to flatten it back to 2D)
    rho_ref_2d = jnp.take_along_axis(rho_3d, k_ref_2d[None, :, :], axis=0)[0]
    
    # --- 3. Find the MLD (Pycnocline Threshold) ---
    # Create a 3D index grid to ensure we only check cells at or below k_ref
    k_indices = jnp.arange(nz)[:, None, None]
    is_valid_search_depth = k_indices >= k_ref_2d[None, :, :]
    
    # Find all cells that exceed the density threshold
    is_dense = (rho_3d - rho_ref_2d[None, :, :]) > delta_rho_mld
    
    # Combine the conditions: Must be below 10m AND exceed threshold
    pycnocline_mask = is_dense & is_valid_search_depth
    
    # Find the FIRST layer in each column where the mask is True
    k_mld_2d = jnp.argmax(pycnocline_mask, axis=0)
    
    # Extract the depth at the TOP of that specific layer
    mld_if_found = jnp.take_along_axis(top_depths_3d, k_mld_2d[None, :, :], axis=0)[0]
    
    # --- 4. Handle Edge Cases & Masking ---
    # Failsafe: Did the column actually hit the pycnocline, or is it fully mixed?
    hit_pycnocline = jnp.any(pycnocline_mask, axis=0)
    
    # If fully mixed, return the total depth of the water column
    total_water_depth = jnp.sum(jnp.where(dz_3d > 1e-6, dz_3d, 0.0), axis=0)
    mld_raw = jnp.where(hit_pycnocline, mld_if_found, total_water_depth)
    
    # Land check: Mask out columns with no water or NaN surface densities
    is_land = (dz_3d[0] < 1e-6) | jnp.isnan(rho_3d[0])
    mld_final = jnp.where(is_land, jnp.nan, mld_raw)
    
    return mld_final

@jax.jit
def calc_o2_flux(o2_surf, o2_saturation, temp_surf, wind_speed, ice_fraction, dz_surf, dt_step):
    """
    Calculates the air-sea flux of Dissolved Oxygen for the surface layer (k=0).
    Fully vectorized for JAX.
    """
    
    # --- 1. Schmidt Number (Wanninkhof 2014) ---
    sc_o2 = (1920.4 
             - 135.6 * temp_surf 
             + 5.2122 * (temp_surf**2) 
             - 0.10939 * (temp_surf**3) 
             + 0.00093777 * (temp_surf**4))
             
    # Safety clamp across the entire array
    sc_o2 = jnp.maximum(sc_o2, 1.0) 
    
    # --- 2. Piston Velocity (kw) ---
    kw_cm_hr = 0.251 * (wind_speed**2) * jnp.power((sc_o2 / 660.0), -0.5)
    
    # Convert kw from cm/hr to m/s
    kw_m_s = kw_cm_hr * (1.0 / 100.0) * (1.0 / 3600.0)
    
    # --- 3. Calculate Flux ---
    # Scale by open water fraction so solid ice prevents gas exchange
    open_water_fraction = jnp.maximum(0.0, 1.0 - ice_fraction)
    flux = kw_m_s * (o2_saturation - o2_surf) * open_water_fraction
    
    # --- 4. Apply Flux & Mask Land ---
    # Create a safe divisor to prevent division-by-zero on land pixels
    dz_safe = jnp.where(dz_surf > 1e-6, dz_surf, 1.0)
    
    # Calculate the raw updated oxygen
    raw_updated = o2_surf + (flux / dz_safe) * dt_step
    
    # Apply the final land mask: if it's ocean, keep the update; if land, return the original
    o2_updated = jnp.where(dz_surf > 1e-6, raw_updated, o2_surf)

    return o2_updated