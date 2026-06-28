import numpy as np
from numba import njit, prange
import math
import taichi as ti

@njit(fastmath=True)
def haversine_dist(lon1, lat1, lon2, lat2):
    """
    Calculates distance (meters) between two points on Earth.
    Inputs in DEGREES.
    """
    R = 6371000.0 # Earth Radius (m)
    
    # Convert to Radians
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    
    a = np.sin(dphi/2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam/2.0)**2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    
    return R * c


@njit(parallel=True, fastmath=True)
def calculate_metrics_curvilinear(lon_2d, lat_2d):
    """
    Calculates dx and dy (meters) for a 2D curvilinear grid.
    dx[j,i] is the distance to the EAST neighbor (i+1).
    dy[j,i] is the distance to the NORTH neighbor (j+1).
    """
    ny, nx = lon_2d.shape
    dx = np.zeros((ny, nx))
    dy = np.zeros((ny, nx))
    
    # 1. Calculate total number of horizontal grid points
    n_points = ny * nx
    
    # 2. Single flattened loop for OpenMP thread distribution
    for p in prange(n_points):
        # 3. Reconstruct 2D spatial indices (j, i)
        j = p // nx
        i = p % nx
        
        # --- DX (Distance along I-axis) ---
        if i < nx - 1:
            dist_x = haversine_dist(lon_2d[j, i], lat_2d[j, i], 
                                    lon_2d[j, i+1], lat_2d[j, i+1])
            dx[j, i] = dist_x
        else:
            # COMPUTE directly instead of copying to prevent parallel race conditions
            dist_x = haversine_dist(lon_2d[j, i-1], lat_2d[j, i-1], 
                                    lon_2d[j, i], lat_2d[j, i])
            dx[j, i] = dist_x
            
        # --- DY (Distance along J-axis) ---
        if j < ny - 1:
            dist_y = haversine_dist(lon_2d[j, i], lat_2d[j, i], 
                                    lon_2d[j+1, i], lat_2d[j+1, i])
            dy[j, i] = dist_y
        else:
            # COMPUTE directly, and fixed index to j-1 (Southern neighbor)
            dist_y = haversine_dist(lon_2d[j-1, i], lat_2d[j-1, i], 
                                    lon_2d[j, i], lat_2d[j, i])
            dy[j, i] = dist_y
            
        # --- Safety Clamp ---
        # Done inside the loop to avoid a second memory pass over the full array
        if dx[j, i] < 1.0:
            dx[j, i] = 1.0
        if dy[j, i] < 1.0:
            dy[j, i] = 1.0
            
    return dx, dy

def calculate_grid_metrics(ds):
    R_EARTH = 6371000.0
    lat = ds['lat'].values
    lon = ds['lon'].values
    if lat.ndim == 1: lon_2d, lat_2d = np.meshgrid(lon, lat)
    else: lon_2d, lat_2d = lon, lat
    
    lat_rad = np.radians(lat_2d)
    lon_rad = np.radians(lon_2d)
    dlat = np.gradient(lat_rad, axis=0)
    dy = R_EARTH * dlat
    dlon = np.gradient(lon_rad, axis=1)
    dx = R_EARTH * np.cos(lat_rad) * dlon
    return np.abs(dx), np.abs(dy)

def calculate_dz_from_centers(depths_1d):
    nz = len(depths_1d)
    interfaces = np.zeros(nz + 1)
    interfaces[0] = 0.0
    for k in range(1, nz): interfaces[k] = 0.5 * (depths_1d[k-1] + depths_1d[k])
    interfaces[nz] = depths_1d[-1] + (depths_1d[-1] - interfaces[nz-1])
    return np.diff(interfaces)


@njit(parallel=True, fastmath=True)
def calculate_w(u, v, dz_3d, dx, dy):
    """
    Calculates W integrating Surface -> Bottom (j, i, k order).
    Includes Linear Correction to force W_bottom = 0.
    """
    nz, ny, nx = u.shape
    w = np.zeros((nz + 1, ny, nx))

    # 1. Calculate the exact number of inner domain points
    n_j = ny - 2  # Equivalent to range(1, ny - 1)
    n_i = nx - 2  # Equivalent to range(1, nx - 1)
    n_points = n_j * n_i

    # 2. Single flattened loop for OpenMP thread distribution
    for p in prange(n_points):
        # 3. Reconstruct spatial indices with a +1 offset to skip boundary walls
        j = 1 + (p // n_i)
        i = 1 + (p % n_i)
            
        # Skip land columns
        if dz_3d[0, j, i] <= 1e-6:
            continue

        dx_c = dx[j, i]
        dy_c = dy[j, i]
        current_w = 0.0
        
        # --- PHASE 1: INTEGRATE DOWNWARD ---
        for k in range(nz):
            dz_c = dz_3d[k, j, i]
            
            # If we hit the seafloor inside the column
            if dz_c <= 1e-6:
                w[k+1, j, i] = 0.0
                continue

            # Face Velocities
            # (Check neighbors for land boundaries)
            u_west  = 0.5 * (u[k, j, i-1] + u[k, j, i]) if dz_3d[k, j, i-1] > 1e-6 else 0.0
            u_east  = 0.5 * (u[k, j, i] + u[k, j, i+1]) if dz_3d[k, j, i+1] > 1e-6 else 0.0
            v_south = 0.5 * (v[k, j-1, i] + v[k, j, i]) if dz_3d[k, j-1, i] > 1e-6 else 0.0
            v_north = 0.5 * (v[k, j, i] + v[k, j+1, i]) if dz_3d[k, j+1, i] > 1e-6 else 0.0
            
            # Divergence
            div_h = ((u_east - u_west) / dx_c) + ((v_north - v_south) / dy_c)
            
            # Update W (Accumulate)
            current_w = current_w + (div_h * dz_c)
            w[k+1, j, i] = current_w

        # --- PHASE 2: LINEAR CORRECTION ---
        # The bottom value (w[nz]) should be 0. If not, remove the error.
        bottom_error = w[nz, j, i]
        
        if abs(bottom_error) > 1e-10:
            for k in range(1, nz + 1):
                # Linearly scale the correction from Surface (0) to Bottom (1)
                weight = float(k) / float(nz)
                w[k, j, i] -= bottom_error * weight
                
    return w


@njit(parallel=True, fastmath=True)
def calculate_w_bottom_up(u, v, dz_3d, dx, dy):
    """
    Calculates W integrating Bottom -> Surface.
    Bypasses the need for SSH time derivatives.
    W at the seafloor is strictly forced to 0.0.
    """
    nz, ny, nx = u.shape
    
    # w array has nz + 1 levels (faces of the cells)
    # w[nz] is the absolute bottom, w[0] is the surface
    w = np.zeros((nz + 1, ny, nx))

    n_j = ny - 2  
    n_i = nx - 2  
    n_points = n_j * n_i

    for p in prange(n_points):
        j = 1 + (p // n_i)
        i = 1 + (p % n_i)
            
        # Skip land columns entirely
        if dz_3d[0, j, i] <= 1e-6:
            continue

        dx_c = dx[j, i]
        dy_c = dy[j, i]
        
        # Start at the seafloor with 0 velocity
        current_w = 0.0
        
        # --- INTEGRATE UPWARD (k goes from nz-1 down to 0) ---
        for k in range(nz - 1, -1, -1):
            dz_c = dz_3d[k, j, i]
            
            # If we are below the seafloor in a stepped-bathymetry model
            if dz_c <= 1e-6:
                w[k, j, i] = 0.0
                continue

            # Face Velocities
            u_west  = 0.5 * (u[k, j, i-1] + u[k, j, i]) if dz_3d[k, j, i-1] > 1e-6 else 0.0
            u_east  = 0.5 * (u[k, j, i] + u[k, j, i+1]) if dz_3d[k, j, i+1] > 1e-6 else 0.0
            v_south = 0.5 * (v[k, j-1, i] + v[k, j, i]) if dz_3d[k, j-1, i] > 1e-6 else 0.0
            v_north = 0.5 * (v[k, j, i] + v[k, j+1, i]) if dz_3d[k, j+1, i] > 1e-6 else 0.0
            
            # Horizontal Divergence
            div_h = ((u_east - u_west) / dx_c) + ((v_north - v_south) / dy_c)
            
            # Update W: W_top = W_bottom - (Divergence * dz)
            current_w = current_w - (div_h * dz_c)
            
            # Assign to the TOP face of the current cell
            w[k, j, i] = current_w
                
    return w


@njit(parallel=True, fastmath=True)
def calculate_w_rigid_lid(u, v, dz_3d, dx, dy):
    """
    Top-Down integration with linear correction.
    Properly detects actual bathymetry (k_bottom) to force
    seafloor W to exactly 0.0 in shallow regions.
    """
    nz, ny, nx = u.shape
    w = np.zeros((nz + 1, ny, nx))

    n_j = ny - 2  
    n_i = nx - 2  
    n_points = n_j * n_i

    for p in prange(n_points):
        j = 1 + (p // n_i)
        i = 1 + (p % n_i)
            
        if dz_3d[0, j, i] <= 1e-6:
            continue

        # --- 1. FIND THE TRUE SEAFLOOR ---
        k_bottom = nz
        for k in range(nz):
            if dz_3d[k, j, i] <= 1e-6:
                k_bottom = k
                break

        dx_c = dx[j, i]
        dy_c = dy[j, i]
        
        # 2. Start at surface with 0 velocity
        current_w = 0.0
        
        # 3. Integrate Downward ONLY to the true seafloor
        for k in range(k_bottom):
            dz_c = dz_3d[k, j, i]

            u_west  = 0.5 * (u[k, j, i-1] + u[k, j, i]) if dz_3d[k, j, i-1] > 1e-6 else 0.0
            u_east  = 0.5 * (u[k, j, i] + u[k, j, i+1]) if dz_3d[k, j, i+1] > 1e-6 else 0.0
            v_south = 0.5 * (v[k, j-1, i] + v[k, j, i]) if dz_3d[k, j-1, i] > 1e-6 else 0.0
            v_north = 0.5 * (v[k, j, i] + v[k, j+1, i]) if dz_3d[k, j+1, i] > 1e-6 else 0.0
            
            div_h = ((u_east - u_west) / dx_c) + ((v_north - v_south) / dy_c)
            
            current_w = current_w + (div_h * dz_c)
            w[k+1, j, i] = current_w

        # --- 4. TRUE LINEAR CORRECTION ---
        # The true bottom value must be 0. 
        bottom_error = w[k_bottom, j, i]
        
        if abs(bottom_error) > 1e-10:
            for k in range(1, k_bottom + 1):
                # Weight increases from 0 (surface) to 1 (seafloor)
                weight = float(k) / float(k_bottom)
                w[k, j, i] -= bottom_error * weight
                
    return w


@ti.kernel
def _advection_neumann_kernel(
    tracer: ti.types.ndarray(dtype=ti.f32),
    tracer_new: ti.types.ndarray(dtype=ti.f32),
    u: ti.types.ndarray(dtype=ti.f32),
    v: ti.types.ndarray(dtype=ti.f32),
    w: ti.types.ndarray(dtype=ti.f32),
    dz: ti.types.ndarray(dtype=ti.f32),
    dx: ti.types.ndarray(dtype=ti.f32),
    dy: ti.types.ndarray(dtype=ti.f32),
    dt: ti.f32,  # Also force the scalar dt to 32-bit
    is_global: ti.i32
):
    nz = tracer.shape[0]
    ny = tracer.shape[1]
    nx = tracer.shape[2]
    
    # Taichi automatically parallelizes this 3D loop across thousands of GPU cores
    for k, j, i in ti.ndrange(nz, ny, nx):
        
        # 1. Replicate the Numba spatial boundary masking (i_start, i_end, n_j)
        is_valid_j = (j >= 1) and (j < ny - 1)
        is_valid_i = True
        if is_global == 0:
            is_valid_i = (i >= 1) and (i < nx - 1)
            
        if is_valid_j and is_valid_i and dz[k, j, i] > 1e-6:
            dx_c = dx[j, i]
            dy_c = dy[j, i]
            dz_c = dz[k, j, i]
            c = tracer[k, j, i]

            # --- 1. Horizontal ---
            u_val = u[k, j, i]
            v_val = v[k, j, i]
            
            i_up = i - 1 if u_val > 0.0 else i + 1
            j_up = j - 1 if v_val > 0.0 else j + 1
            
            if is_global == 1:
                if i_up < 0:
                    i_up = nx - 1
                elif i_up >= nx:
                    i_up = 0
            
            # Safe neighbor assignment
            c_up_x = tracer[k, j, i_up] if dz[k, j, i_up] > 1e-6 else c
            c_up_y = tracer[k, j_up, i] if dz[k, j_up, i] > 1e-6 else c
            
            term_x = ti.abs(u_val) * (c - c_up_x) / dx_c
            term_y = ti.abs(v_val) * (c - c_up_y) / dy_c

            # --- 2. Vertical ---
            w_val = 0.5 * (w[k, j, i] + w[k+1, j, i])
            
            # Nested IFs prevent GPU out-of-bounds indexing crashes
            c_below = c
            if k < nz - 1:
                if dz[k+1, j, i] > 1e-6:
                    c_below = tracer[k+1, j, i]
                    
            c_above = c
            if k > 0:
                if dz[k-1, j, i] > 1e-6:
                    c_above = tracer[k-1, j, i]

            term_z = 0.0
            if w_val > 0.0:
                term_z = w_val * (c - c_below) / dz_c
            else:
                term_z = w_val * (c_above - c) / dz_c

            # --- 3. Total Change ---
            raw_c = c - dt * (term_x + term_y + term_z)
            tracer_new[k, j, i] = ti.max(0.0, raw_c)

# --- PYTHON WRAPPER ---
# Keep your API identical to the Numba version!
def advection_neumann(tracer, u, v, w, dz, dt, dx, dy, is_global=False):
    # Convert all incoming arrays to float32 for the GPU
    tracer_f32 = tracer.astype(np.float32)
    u_f32 = u.astype(np.float32)
    v_f32 = v.astype(np.float32)
    w_f32 = w.astype(np.float32)
    dz_f32 = dz.astype(np.float32)
    dx_f32 = dx.astype(np.float32)
    dy_f32 = dy.astype(np.float32)
    
    # Create the output array as float32
    tracer_new_f32 = tracer_f32.copy()
    
    # Run the GPU kernel
    _advection_neumann_kernel(
        tracer_f32, tracer_new_f32, u_f32, v_f32, w_f32, dz_f32, dx_f32, dy_f32, float(dt), 1 if is_global else 0
    )
    
    return tracer_new_f32

@ti.kernel
def _calculate_full_kz_kernel(
    mld_2d: ti.types.ndarray(dtype=ti.f32),
    rho_3d: ti.types.ndarray(dtype=ti.f32),
    dz_3d: ti.types.ndarray(dtype=ti.f32),
    kz_3d: ti.types.ndarray(dtype=ti.f32),
    k_mld_max: ti.f32,
    k_bg_min: ti.f32,
    k_deep_max: ti.f32
):
    nz = dz_3d.shape[0]
    ny = dz_3d.shape[1]
    nx = dz_3d.shape[2]
    
    g = 9.81
    rho_0 = 1035.0
    
    # Thread over the 2D surface grid (each thread gets one column)
    for j, i in ti.ndrange(ny, nx):
        mld = mld_2d[j, i]
        
        # Note: Taichi uses ti.math.isnan() instead of np.isnan() inside kernels
        if ti.math.isnan(mld) or dz_3d[0, j, i] < 1e-6:
            for k in range(nz):
                kz_3d[k, j, i] = ti.math.nan
            continue
            
        current_depth = 0.0
        
        for k in range(nz):
            dz = dz_3d[k, j, i]
            current_depth += dz
            
            # --- REGION 1: INSIDE MLD ---
            if current_depth <= mld:
                sigma = current_depth / mld
                shape = sigma * (1.0 - sigma)**2
                kz_3d[k, j, i] = k_bg_min + (k_mld_max * 6.75 * shape)
            
            # --- REGION 2: BELOW MLD ---
            else:
                N2 = 1e-7
                if k < nz - 1:
                    drho = rho_3d[k+1, j, i] - rho_3d[k, j, i]
                    dz_eff = 0.5 * (dz_3d[k, j, i] + dz_3d[k+1, j, i])
                    local_N2 = (g / rho_0) * (drho / dz_eff)
                    N2 = ti.max(local_N2, 1e-7)
                
                k_deep = 1e-11 / ti.math.sqrt(N2)
                
                kz_3d[k, j, i] = ti.min(ti.max(k_deep, k_bg_min), k_deep_max)


def calculate_full_kz(mld_2d, rho_3d, dz_3d, k_mld_max=1e-2, k_bg_min=1e-9, k_deep_max=1e-5):
    # Downcast to 32-bit for the Metal GPU
    mld_f32 = mld_2d.astype(np.float32)
    rho_f32 = rho_3d.astype(np.float32)
    dz_f32  = dz_3d.astype(np.float32)
    kz_out_f32 = np.zeros_like(dz_f32)
    
    _calculate_full_kz_kernel(
        mld_f32, rho_f32, dz_f32, kz_out_f32,
        float(k_mld_max), float(k_bg_min), float(k_deep_max)
    )
    
    return kz_out_f32

@ti.kernel
def _diffusion_robust_kernel(
    tracer: ti.types.ndarray(dtype=ti.f32),
    k_z: ti.types.ndarray(dtype=ti.f32),
    dz_3d: ti.types.ndarray(dtype=ti.f32),
    tracer_out: ti.types.ndarray(dtype=ti.f32),
    dt: ti.f32
):
    nz = tracer.shape[0]
    ny = tracer.shape[1]
    nx = tracer.shape[2]
    epsilon = 1e-6
    
    # Thread over the 2D surface grid
    for j, i in ti.ndrange(ny, nx):
        
        if ti.math.isnan(tracer[0, j, i]):
            for k in range(nz):
                tracer_out[k, j, i] = ti.math.nan
            continue

        # Surface flux is always zero (rigid lid / no atmospheric loss)
        flux_top = 0.0
        
        # Sequentially walk down the column
        for k in range(nz):
            vol = dz_3d[k, j, i]
            flux_bottom = 0.0
            
            # 1. Calculate flux at the bottom interface of this cell
            if k < nz - 1:
                delta_z = 0.5 * (dz_3d[k, j, i] + dz_3d[k+1, j, i])
                
                if delta_z >= epsilon:
                    k_face = 0.5 * (k_z[k, j, i] + k_z[k+1, j, i])
                    k_max = 0.45 * (delta_z * delta_z) / dt
                    
                    k_eff = k_max if k_face > k_max else k_face
                    
                    grad = (tracer[k+1, j, i] - tracer[k, j, i]) / delta_z
                    flux_bottom = k_eff * grad

            # 2. Update the cell using Flux In (bottom) and Flux Out (top)
            if vol < epsilon:
                tracer_out[k, j, i] = tracer[k, j, i]
            else:
                trend = (flux_bottom - flux_top) / vol
                tracer_out[k, j, i] = tracer[k, j, i] + dt * trend
                
            # 3. Pass this cell's bottom flux down to become the next cell's top flux
            flux_top = flux_bottom


def diffusion_robust(tracer, k_z, dz_3d, dt):
    tracer_f32 = tracer.astype(np.float32)
    kz_f32 = k_z.astype(np.float32)
    dz_f32 = dz_3d.astype(np.float32)
    tracer_out_f32 = tracer_f32.copy()
    
    _diffusion_robust_kernel(
        tracer_f32, kz_f32, dz_f32, tracer_out_f32, float(dt)
    )
    
    return tracer_out_f32


def create_restoring_weights(water_mask, sponge_width, tau_lateral, tau_bottom, dt, is_global=False):
    """
    Creates a 3D rate array (1/sec) for restoring.
    It combines Lateral Sponge (Fast) and Seafloor Restoring (Slow).
    """
    nz, ny, nx = water_mask.shape
    
    # 1. Initialize Rates with Zeros
    restore_rate = np.zeros((nz, ny, nx))
    
    # Define Rates
    rate_lateral = 1.0 / tau_lateral
    rate_bottom  = 1.0 / tau_bottom
    
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

    # --- B. SEAFLOOR RESTORING (The Fix) ---
    # We iterate over 2D surface to find the deepest wet cell k
    # (Looping over 2D surface is fast enough for setup)
    for j in prange(ny):
        for i in range(nx):
            # Extract the column
            col = water_mask[:, j, i]
            
            # Skip if Land (all False)
            if not np.any(col):
                continue
                
            # Find the deepest wet index
            # np.where returns indices where condition is True. We take the last one.
            k_bot = np.where(col)[0][-1]
            
            # Apply Bottom Rate
            # CRITICAL: Do we override the Sponge?
            # Rule: If the bottom is effectively part of the "Sponge Wall" (e.g. shallow shelf),
            # we should keep the FASTER rate (Lateral).
            # If it's the deep ocean floor, we add the SLOWER rate (Bottom).
            
            # Let's take the maximum of existing sponge rate vs bottom rate
            # This ensures fast restoring at boundaries, slow restoring at deep bottom.
            restore_rate[k_bot, j, i] = np.max([restore_rate[k_bot, j, i], rate_bottom])

    # Multiply by DT to get the "nudge fraction" per step
    # Result is a 3D array of alpha values: C_new = C_old + alpha * (C_target - C_old)
    nudge_coeff = restore_rate * dt
    
    # Stability Check: alpha cannot exceed 1.0
    nudge_coeff = np.minimum(nudge_coeff, 1.0)
    
    # Mask out land finally (just to be safe)
    nudge_coeff = np.where(water_mask, nudge_coeff, 0.0)
    
    return nudge_coeff

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


@njit(parallel=True, fastmath=True)
def calculate_mld(rho_3d, dz_3d, delta_rho_mld):
    """
    Calculates Mixed Layer Depth (m) using the standard 10m reference depth.
    Bypasses thin surface freshwater/diurnal lenses.
    """
    nz, ny, nx = rho_3d.shape
    mld_2d = np.zeros((ny, nx))
    
    # 1. Calculate total number of horizontal grid points
    n_points = ny * nx
    
    # 2. Single flattened loop for maximum OpenMP thread distribution
    for p in prange(n_points):
        # 3. Reconstruct 2D spatial indices (j, i) from the 1D index (p)
        j = p // nx
        i = p % nx
        
        # Land Check
        if dz_3d[0, j, i] < 1e-6 or np.isnan(rho_3d[0, j, i]):
            mld_2d[j, i] = np.nan
            continue
        
        # 1. Find the reference layer (closest to 10m depth)
        depth_accum = 0.0
        k_ref = 0
        for k in range(nz):
            depth_accum += dz_3d[k, j, i]
            if depth_accum >= 10.0:
                k_ref = k
                break
                
        rho_ref = rho_3d[k_ref, j, i]
        
        # 2. Calculate MLD checking against the 10m reference
        current_depth = 0.0
        for k in range(nz):
            current_depth += dz_3d[k, j, i]
            
            # Only check for the threshold once we are at or below the 10m reference
            if k >= k_ref:
                # If the water becomes significantly heavier than the 10m water
                if (rho_3d[k, j, i] - rho_ref) > delta_rho_mld:
                    # We hit the pycnocline! Back up to the top of this layer and break.
                    current_depth -= dz_3d[k, j, i]
                    break
                    
        # Failsafe: if the whole column is mixed, it will just return the domain bottom
        mld_2d[j, i] = current_depth
        
    return mld_2d


@njit(parallel=True, fastmath=True)
def calc_o2_flux(o2_surf, o2_saturation, temp_surf, wind_speed, ice_fraction, dz_surf, dt_step):
    """
    Calculates the air-sea flux of Dissolved Oxygen for the surface layer (k=0).
    Uses Wanninkhof (2014) for piston velocity with dynamic Schmidt numbers, 
    and applies a sea ice mask.
    
    Args:
        o2_surf: 2D array of surface O2 (mmol/m3)
        o2_saturation: 2D array of O2 saturation (mmol/m3) calculated via gsw
        temp_surf: 2D array of Sea Surface Temperature (Celsius)
        wind_speed: 2D array of 10m wind speed (m/s)
        ice_fraction: 2D array of sea ice concentration (0.0 to 1.0)
        dz_surf: 2D array of surface grid cell thickness (m)
        dt_step: Time step (seconds)
        
    Returns:
        o2_updated: 2D array of updated surface O2 concentrations
    """
    ny, nx = o2_surf.shape
    
    # Use .copy() to preserve land values/masks safely
    # (Doing this outside the prange loop is completely safe and fast)
    o2_updated = o2_surf.copy()
    
    # 1. Calculate total number of horizontal grid points
    n_points = ny * nx
    
    # 2. Single flattened loop for maximum OpenMP thread distribution
    for p in prange(n_points):
        # 3. Reconstruct 2D spatial indices (j, i) from the 1D index (p)
        j = p // nx
        i = p % nx
        
        if dz_surf[j, i] < 1e-6:
            continue # Skip land
            
        o2_local = o2_surf[j, i]
        sat_local = o2_saturation[j, i]
        temp_local = temp_surf[j, i]
        wind_local = wind_speed[j, i]
        ice_local = ice_fraction[j, i]
        
        # 1. Calculate Schmidt number for O2 (Wanninkhof 2014)
        # Valid for seawater from -2 to 40 Celsius
        sc_o2 = (1920.4 
                 - 135.6 * temp_local 
                 + 5.2122 * (temp_local**2) 
                 - 0.10939 * (temp_local**3) 
                 + 0.00093777 * (temp_local**4))
                 
        sc_o2 = max(sc_o2, 1.0) # Safety clamp to prevent negative/zero division
        
        # 2. Calculate Piston Velocity (kw) in cm/hr
        kw_cm_hr = 0.251 * (wind_local**2) * ((sc_o2 / 660.0)**-0.5)
        
        # Convert kw from cm/hr to m/s
        kw_m_s = kw_cm_hr * (1.0 / 100.0) * (1.0 / 3600.0)
        
        # 3. Calculate Flux (mmol O2 / m2 / s)
        # Scale by open water fraction so solid ice prevents gas exchange
        open_water_fraction = max(0.0, 1.0 - ice_local)
        flux = kw_m_s * (sat_local - o2_local) * open_water_fraction
        
        # 4. Apply flux to the surface layer concentration
        o2_updated[j, i] = o2_local + (flux / dz_surf[j, i]) * dt_step

    return o2_updated