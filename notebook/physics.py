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
def diffusion(tracer, kh, dz, dt):
    """
    Optimized: Removes internal array allocation O(N) -> O(1).
    """
    nz, ny, nx = tracer.shape
    tracer_out = tracer.copy()
    
    for j in prange(ny):
        for i in range(nx):
            if dz[0, j, i] <= 1e-6: continue

            # RUNNING FLUX VARIABLES
            # flux_upper is the flux coming from above (k-1 -> k)
            # At surface (k=0), flux from above is 0.
            flux_upper = 0.0 
            
            for k in range(nz):
                # 1. Compute flux at the bottom interface (k -> k+1)
                flux_lower = 0.0
                
                # Check if we are at the last face
                if k < nz - 1:
                    dz_curr = dz[k, j, i]
                    dz_next = dz[k+1, j, i]
                    
                    if dz_next > 1e-6:
                        # Diffusivity & Distance
                        k_val = 0.5 * (kh[k, j, i] + kh[k+1, j, i])
                        dist = 0.5 * (dz_curr + dz_next)
                        
                        # Gradient (Current - Next)
                        dC = tracer_out[k, j, i] - tracer_out[k+1, j, i]
                        flux_lower = k_val * dC / dist

                # 2. Update Current Cell
                # Change = Flux_In (Upper) - Flux_Out (Lower)
                if dz[k, j, i] > 1e-6:
                    tracer_out[k, j, i] += dt * (flux_upper - flux_lower) / dz[k, j, i]
                
                # 3. Pass the torch: Lower flux becomes next cell's Upper flux
                flux_upper = flux_lower

    return tracer_out

@njit(parallel=True, fastmath=True)
def diffusion_explicit_smart(tracer, k_curr, dz_3d, dt):
    """
    Explicit diffusion with dynamic stability clipping.
    Max K is calculated locally based on grid thickness dz.
    """
    nz, ny, nx = tracer.shape
    tracer_out = np.empty_like(tracer)
    
    # Safety factor (must be < 0.5)
    alpha = 0.45 
    
    for j in prange(ny):
        for i in range(nx):
            # 1. Skip Land
            if np.isnan(tracer[0, j, i]):
                tracer_out[:, j, i] = np.nan
                continue
                
            # 2. Local Column Loop
            # We solve the flux divergence: dC/dt = d/dz(K dC/dz)
            flux = np.zeros(nz + 1) # Fluxes at interfaces
            
            for k in range(nz - 1): # Interfaces 0 to nz-2
                # Distance between center k and k+1
                delta_z = 0.5 * (dz_3d[k, j, i] + dz_3d[k+1, j, i])
                
                # Interface K (average)
                k_face = 0.5 * (k_curr[k, j, i] + k_curr[k+1, j, i])
                
                # --- DYNAMIC CLIPPING (The Magic Fix) ---
                # Stability Condition: K * dt / dz^2 < 0.5
                # Max allowed K for this specific interface
                k_max = alpha * (delta_z**2) / dt
                
                # Clip K locally!
                k_safe = min(k_face, k_max)
                
                # Calculate Flux (Diffusive Law)
                # Flux is defined POSITIVE UP (from k+1 to k)
                gradient = (tracer[k+1, j, i] - tracer[k, j, i]) / delta_z
                flux[k+1] = k_safe * gradient

            # Boundary Conditions (No Flux)
            flux[0]  = 0.0 # Surface
            flux[nz] = 0.0 # Bottom
            
            # 3. Update Tracer
            for k in range(nz):
                # Volume of cell
                vol = dz_3d[k, j, i]
                # Divergence of flux
                # (Flux entering from bottom - Flux leaving top)
                # Note indices: flux[k] is top, flux[k+1] is bottom
                tracer_out[k, j, i] = tracer[k, j, i] + (dt / vol) * (flux[k+1] - flux[k])
                
    return tracer_out

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

@njit(parallel=True, fastmath=True)
def diffusion_implicit(tracer, k_z, dz, dt):
    """
    Solves Vertical Diffusion using the Implicit (Backward Euler) method.
    Unconditionally stable for any K and dt.
    Solves the Tridiagonal Matrix using the Thomas Algorithm.
    """
    nz, ny, nx = tracer.shape
    tracer_out = np.empty_like(tracer)
    
    # Loop over every water column (Parallelized)
    for j in prange(ny):
        for i in range(nx):
            
            # 1. Check for land (if surface is NaN or 0)
            if np.isnan(tracer[0, j, i]):
                tracer_out[:, j, i] = np.nan
                continue

            # 2. Setup Tridiagonal Matrix Arrays
            # Equation: -a*C[k-1] + b*C[k] - c*C[k+1] = d
            a = np.zeros(nz) # Lower diagonal
            b = np.zeros(nz) # Main diagonal
            c = np.zeros(nz) # Upper diagonal
            d = np.zeros(nz) # RHS (Current Tracer)
            
            # We need the interface diffusion coefficients
            # simple average for interface k-1/2
            
            for k in range(nz):
                d[k] = tracer[k, j, i] # Right Hand Side is current state
                
                # --- Terms for Surface (k=0) ---
                if k == 0:
                    # Flux from below (k to k+1)
                    # K at interface 0.5 (between 0 and 1)
                    # We approximate K_interface as K[k] for simplicity or average
                    k_down = 0.5 * (k_z[k, j, i] + k_z[k+1, j, i])
                    
                    alpha = k_down * dt / (dz[k, j, i] * dz[k+1, j, i]) # Simplified dz handling
                    # Ideally: dz[k] * distance_between_centers
                    # Better: dist = 0.5*(dz[k] + dz[k+1])
                    dist_down = 0.5 * (dz[k, j, i] + dz[k+1, j, i])
                    coeff_down = (k_down * dt) / (dz[k, j, i] * dist_down)
                    
                    b[k] = 1.0 + coeff_down
                    c[k] = coeff_down
                    
                # --- Terms for Bottom (k=nz-1) ---
                elif k == nz - 1:
                    # Flux from above
                    k_up = 0.5 * (k_z[k, j, i] + k_z[k-1, j, i])
                    dist_up = 0.5 * (dz[k, j, i] + dz[k-1, j, i])
                    coeff_up = (k_up * dt) / (dz[k, j, i] * dist_up)
                    
                    a[k] = coeff_up
                    b[k] = 1.0 + coeff_up
                    
                # --- Terms for Interior ---
                else:
                    # Downward (k to k+1)
                    k_down = 0.5 * (k_z[k, j, i] + k_z[k+1, j, i])
                    dist_down = 0.5 * (dz[k, j, i] + dz[k+1, j, i])
                    coeff_down = (k_down * dt) / (dz[k, j, i] * dist_down)
                    
                    # Upward (k to k-1)
                    k_up = 0.5 * (k_z[k, j, i] + k_z[k-1, j, i])
                    dist_up = 0.5 * (dz[k, j, i] + dz[k-1, j, i])
                    coeff_up = (k_up * dt) / (dz[k, j, i] * dist_up)
                    
                    a[k] = coeff_up
                    c[k] = coeff_down
                    b[k] = 1.0 + coeff_up + coeff_down

            # 3. Thomas Algorithm (TDMA) Solver
            # Forward elimination
            c[0] = c[0] / b[0]
            d[0] = d[0] / b[0]
            
            for k in range(1, nz):
                temp = b[k] - a[k] * c[k-1]
                if temp == 0.0: temp = 1e-20 # Safety
                c[k] = c[k] / temp
                d[k] = (d[k] - a[k] * d[k-1]) / temp
                
            # Backward substitution
            tracer_out[nz-1, j, i] = d[nz-1]
            for k in range(nz-2, -1, -1):
                tracer_out[k, j, i] = d[k] - c[k] * tracer_out[k+1, j, i]

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