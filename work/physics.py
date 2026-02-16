import numpy as np
from numba import njit, prange

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
    
    for j in prange(ny):
        for i in range(nx):
            # --- DX (Distance along I-axis) ---
            # Distance from (j,i) to (j, i+1)
            # Boundary: Use previous cell's width at the edge
            if i < nx - 1:
                dist = haversine_dist(lon_2d[j, i], lat_2d[j, i], 
                                      lon_2d[j, i+1], lat_2d[j, i+1])
                dx[j, i] = dist
            else:
                dx[j, i] = dx[j, i-1] # Copy edge
                
            # --- DY (Distance along J-axis) ---
            # Distance from (j,i) to (j+1, i)
            if j < ny - 1:
                dist = haversine_dist(lon_2d[j, i], lat_2d[j, i], 
                                      lon_2d[j+1, i], lat_2d[j+1, i])
                dy[j, i] = dist
            else:
                dy[j, i] = dy[j, i-1] # Copy edge
                
    # Safety: Avoid division by zero on land or weird points
    dx = np.maximum(dx, 1.0)
    dy = np.maximum(dy, 1.0)
    
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

    # Parallelize over Latitude (J) for best load balancing
    for j in prange(1, ny - 1):
        for i in range(1, nx - 1):
            
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
def advection_neumann(tracer, u, v, w, dz, dt, dx, dy):
    nz, ny, nx = tracer.shape
    tracer_new = np.zeros_like(tracer)
    
    for j in prange(1, ny - 1):
        for k in range(nz):
            for i in range(1, nx - 1):
                if dz[k, j, i] <= 1e-6: continue

                dx_c = dx[j, i]
                dy_c = dy[j, i]
                dz_c = dz[k, j, i]

                # --- 1. Horizontal (Collapsed Logic) ---
                u_val = u[k, j, i]
                v_val = v[k, j, i]
                
                # Identify Neighbors (Neumann Check Inline)
                # If u>0, neighbor is i-1. If u<0, neighbor is i+1.
                i_up = i - 1 if u_val > 0 else i + 1
                j_up = j - 1 if v_val > 0 else j + 1
                
                # Get Values (Check if neighbor is land)
                c = tracer[k, j, i]
                c_up_x = tracer[k, j, i_up] if dz[k, j, i_up] > 1e-6 else c
                c_up_y = tracer[k, j_up, i] if dz[k, j_up, i] > 1e-6 else c
                
                # Generalized Advection Formula: |u| * (Center - Upwind) / dx
                # Note: This effectively computes u * dC/dx
                term_x = np.abs(u_val) * (c - c_up_x) / dx_c
                term_y = np.abs(v_val) * (c - c_up_y) / dy_c

                # --- 2. Vertical (Standard) ---
                w_val = 0.5 * (w[k, j, i] + w[k+1, j, i])
                if k == 0: 
                    term_z = 0.0 if w_val < 0 else w_val * (c - tracer[k+1, j, i]) / dz_c
                elif k == nz - 1:
                    term_z = 0.0 if w_val > 0 else w_val * (tracer[k-1, j, i] - c) / dz_c
                else:
                    # For vertical, we stick to standard logic as indices are hard boundaries
                    if w_val > 0: term_z = w_val * (c - tracer[k+1, j, i]) / dz_c
                    else:         term_z = w_val * (tracer[k-1, j, i] - c) / dz_c

                # --- 3. Total Change ---
                # Since we calculated terms as |u|*(C-Cup)/dx, this IS the advection term.
                # Just subtract it.
                tracer_new[k, j, i] = c - dt * (term_x + term_y + term_z)

    return tracer_new

@njit(parallel=True, fastmath=True)
def diffusion_robust(tracer, k_z, dz_3d, dt):
    """
    Robust Explicit Diffusion.
    1. Checks for dz < epsilon to prevent Divide-By-Zero.
    2. Clamps K_z locally to ensure stability (CFL < 0.5).
    """
    nz, ny, nx = tracer.shape
    tracer_out = np.empty_like(tracer)
    
    # Tiny number to avoid division by zero
    epsilon = 1e-6
    
    for j in prange(ny):
        for i in range(nx):
            # 1. LAND CHECK: If surface is NaN, the whole column is land.
            if np.isnan(tracer[0, j, i]):
                tracer_out[:, j, i] = np.nan
                continue

            # Initialize Fluxes (defined at interfaces)
            # flux[k] is flux across top interface of cell k
            flux = np.zeros(nz + 1)

            # --- CALCULATE FLUXES (Interfaces) ---
            for k in range(nz - 1): # Interfaces 0, 1, ..., nz-2
                # Interface is between cell k (Top) and k+1 (Bottom)
                
                # Effective thickness distance
                delta_z = 0.5 * (dz_3d[k, j, i] + dz_3d[k+1, j, i])
                
                # SAFETY 1: If dz is zero (bottom boundary or weird grid), NO FLUX.
                if delta_z < epsilon:
                    flux[k+1] = 0.0
                    continue

                # Interface K (Arithmetic Mean)
                k_face = 0.5 * (k_z[k, j, i] + k_z[k+1, j, i])
                
                # SAFETY 2: Stability Clamp
                # Max allowed K = 0.5 * dz^2 / dt
                # We use 0.45 for safety margin
                k_max = 0.45 * (delta_z**2) / dt
                
                # If K is too huge for this grid cell, clamp it.
                if k_face > k_max:
                    k_eff = k_max
                else:
                    k_eff = k_face
                
                # Calculate Flux (Positive Upwards)
                # Flux = K * dC/dz
                grad = (tracer[k+1, j, i] - tracer[k, j, i]) / delta_z
                flux[k+1] = k_eff * grad

            # Top and Bottom Boundary Conditions (No Flux)
            flux[0]  = 0.0
            flux[nz] = 0.0
            
            # --- UPDATE TRACER ---
            for k in range(nz):
                vol = dz_3d[k, j, i]
                
                # SAFETY 3: Avoid updating zero-volume cells (land)
                if vol < epsilon:
                    tracer_out[k, j, i] = tracer[k, j, i]
                else:
                    # Divergence: Flux In (Bottom) - Flux Out (Top)
                    # flux[k+1] enters from bottom
                    # flux[k] leaves from top
                    trend = (flux[k+1] - flux[k]) / vol
                    tracer_out[k, j, i] = tracer[k, j, i] + dt * trend
                    
    return tracer_out

def create_restoring_weights(water_mask, sponge_width, tau_lateral, tau_bottom, dt):
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
        
        # Apply to 4 sides
        # Use np.maximum so we don't overwrite if corners overlap
        restore_rate[:, :, i]      = np.maximum(restore_rate[:, :, i], val)       # West
        restore_rate[:, :, -(i+1)] = np.maximum(restore_rate[:, :, -(i+1)], val)  # East
        restore_rate[:, i, :]      = np.maximum(restore_rate[:, i, :], val)       # South
        restore_rate[:, -(i+1), :] = np.maximum(restore_rate[:, -(i+1), :], val)  # North

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
    
    for j in prange(ny):
        for i in range(nx):
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