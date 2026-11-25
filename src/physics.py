import numpy as np
import taichi as ti
import math
import xarray as xr

# ==========================================
# 1. SETUP FUNCTIONS (Pure NumPy - Runs Once)
# ==========================================

def generate_static_grid(ds_template):
    R_EARTH = 6371000.0
    
    # 1. Calculate 1D differences and pad the last value
    dlon_vals = np.abs(np.diff(ds_template['lon'].values))
    dlon_vals = np.append(dlon_vals, dlon_vals[-1])
    
    dlat_vals = np.abs(np.diff(ds_template['lat'].values))
    dlat_vals = np.append(dlat_vals, dlat_vals[-1])
    
    lat_vals = ds_template['lat'].values
    
    # 2. Use meshgrid to generate full 2D grids for math
    # indexing='ij' ensures the output shape is strictly (lat, lon)
    dlat_2d, dlon_2d = np.meshgrid(dlat_vals, dlon_vals, indexing='ij')
    lat_2d, _ = np.meshgrid(lat_vals, dlon_vals, indexing='ij')
    
    # 3. Convert all 2D grids to radians
    dlon_rad = np.radians(dlon_2d)
    dlat_rad = np.radians(dlat_2d)
    lat_rad = np.radians(lat_2d)
    
    # 4. Calculate 2D Area directly (no manual broadcasting needed!)
    area = (R_EARTH**2) * dlat_rad * dlon_rad * np.cos(lat_rad)
    
    # 5. Build the Xarray Dataset
    ds_grid = xr.Dataset(
        data_vars={'area': (['lat', 'lon'], area)},
        coords=ds_template.coords
    )
    
    # 6. Handle Depth / Volume if present
    if 'depth' in ds_template.coords:
        ddep_vals = np.abs(np.diff(ds_template['depth'].values))
        ddep_vals = np.append(ddep_vals, ddep_vals[-1])
        
        ds_grid['dz'] = (['depth'], ddep_vals)
        
        # Note: We still use basic numpy broadcasting for the 3D volume step.
        # Generating a full 3D meshgrid would waste a massive amount of RAM,
        # and simple 1D depth * 2D area broadcasting is standard and safe here.
        volume = area[None, :, :] * ddep_vals[:, None, None]
        
        ds_grid['volume'] = (['depth', 'lat', 'lon'], volume)
        ds_grid['dz'].attrs = {'units': 'm'}
        ds_grid['volume'].attrs = {'units': 'm3'}
        
    ds_grid['area'].attrs = {'units': 'm2'}
    
    return ds_grid

def haversine_dist(lon1, lat1, lon2, lat2):
    R = 6371000.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi/2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam/2.0)**2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return R * c

def calculate_metrics_curvilinear(lon_2d, lat_2d):
    dx_inner = haversine_dist(lon_2d[:, :-1], lat_2d[:, :-1], lon_2d[:, 1:], lat_2d[:, 1:])
    dx = np.concatenate([dx_inner, dx_inner[:, -1:]], axis=1)
    dy_inner = haversine_dist(lon_2d[:-1, :], lat_2d[:-1, :], lon_2d[1:, :], lat_2d[1:, :])
    dy = np.concatenate([dy_inner, dy_inner[-1:, :]], axis=0)
    return np.maximum(dx, 1.0), np.maximum(dy, 1.0)

def calculate_grid_metrics(ds):
    lat = ds['lat'].values
    lon = ds['lon'].values
    if lat.ndim == 1:
        lon_2d, lat_2d = np.meshgrid(lon, lat)
    else:
        lon_2d, lat_2d = lon, lat
        
    R_EARTH = 6371000.0
    lat_rad = np.radians(lat_2d)
    lon_rad = np.radians(lon_2d)
    
    dlat = np.gradient(lat_rad, axis=0)
    dy = R_EARTH * dlat
    dlon = np.gradient(lon_rad, axis=1)
    dx = R_EARTH * np.cos(lat_rad) * dlon
    return np.maximum(np.abs(dx), 1.0), np.maximum(np.abs(dy), 1.0)

def calculate_dz_from_centers(depths_1d):
    midpoints = 0.5 * (depths_1d[:-1] + depths_1d[1:])
    surface = np.array([0.0])
    bottom = np.array([depths_1d[-1] + (depths_1d[-1] - midpoints[-1])])
    interfaces = np.concatenate([surface, midpoints, bottom])
    return np.diff(interfaces)

def create_restoring_weights(water_mask, sponge_width, dt, tau_lateral=86400.0, tau_bottom=86400.0*30, tau_coast=86400.0, is_global=False):
    nz, ny, nx = water_mask.shape
    restore_rate = np.zeros((nz, ny, nx))
    rate_lateral, rate_bottom, rate_coast = 1.0 / tau_lateral, 1.0 / tau_bottom, 1.0 / tau_coast
    
    for i in range(sponge_width):
        val = ((sponge_width - i) / sponge_width) * rate_lateral
        restore_rate[:, i, :] = np.maximum(restore_rate[:, i, :], val)
        restore_rate[:, -(i+1), :] = np.maximum(restore_rate[:, -(i+1), :], val)
        if not is_global:
            restore_rate[:, :, i] = np.maximum(restore_rate[:, :, i], val)
            restore_rate[:, :, -(i+1)] = np.maximum(restore_rate[:, :, -(i+1)], val)

    for j in range(ny):
        for i in range(nx):
            col = water_mask[:, j, i]
            if np.any(col):
                k_bot = np.where(col)[0][-1]
                restore_rate[k_bot, j, i] = np.maximum(restore_rate[k_bot, j, i], rate_bottom)

    coastal_mask_2d = np.zeros_like(water_mask[0], dtype=bool)
    surface_water = water_mask[0]
    coastal_mask_2d[1:, :] |= (~surface_water[:-1, :])
    coastal_mask_2d[:-1, :] |= (~surface_water[1:, :])
    coastal_mask_2d[:, 1:] |= (~surface_water[:, :-1])
    coastal_mask_2d[:, :-1] |= (~surface_water[:, 1:])
    coastal_mask_2d &= surface_water
    
    restore_rate[0] = np.where(coastal_mask_2d, np.maximum(restore_rate[0], rate_coast), restore_rate[0])
    nudge_coeff = np.where(water_mask, np.minimum(restore_rate * dt, 1.0), 0.0)
    return nudge_coeff

@ti.kernel
def apply_restoring_ti(
    tracer: ti.types.ndarray(),
    clim: ti.types.ndarray(),
    nudge: ti.types.ndarray()
):
    """
    Applies climatology restoring 100% in-place. 
    Zero temporary arrays created!
    """
    nz, ny, nx = tracer.shape
    for k, j, i in ti.ndrange(nz, ny, nx):
        # Calculate the difference and apply the nudge weight instantly
        diff = clim[k, j, i] - tracer[k, j, i]
        tracer[k, j, i] += diff * nudge[k, j, i]


# ==========================================
# 2. TAICHI GPU PHYSICS KERNELS
# ==========================================
@ti.func
def van_leer_limiter(r):
    return (r + ti.abs(r)) / (1.0 + ti.abs(r))

@ti.func
def minmod_limiter(r):
    return ti.max(0.0, ti.min(1.0, r))
    
@ti.kernel
def filter_uv_checkerboard_ti(
    u: ti.template(), v: ti.template(), 
    u_temp: ti.template(), v_temp: ti.template(), 
    dz: ti.template()
):
    nz, ny, nx = u.shape
    
    # 1. Copy original values to the temp arrays
    for k, j, i in u:
        u_temp[k, j, i] = u[k, j, i]
        v_temp[k, j, i] = v[k, j, i]

    # 2. Mask-aware smoothing (only average with valid ocean neighbors)
    for k, j, i in ti.ndrange(nz, (1, ny-1), (1, nx-1)):
        if dz[k, j, i] > 1e-6:
            u_sum = u[k, j, i]
            v_sum = v[k, j, i]
            valid_weight = 1.0
            
            # Check West
            if dz[k, j, i-1] > 1e-6:
                u_sum += 0.5 * u[k, j, i-1]
                v_sum += 0.5 * v[k, j, i-1]
                valid_weight += 0.5
                
            # Check East
            if dz[k, j, i+1] > 1e-6:
                u_sum += 0.5 * u[k, j, i+1]
                v_sum += 0.5 * v[k, j, i+1]
                valid_weight += 0.5
                
            # Check South
            if dz[k, j-1, i] > 1e-6:
                u_sum += 0.5 * u[k, j-1, i]
                v_sum += 0.5 * v[k, j-1, i]
                valid_weight += 0.5
                
            # Check North
            if dz[k, j+1, i] > 1e-6:
                u_sum += 0.5 * u[k, j+1, i]
                v_sum += 0.5 * v[k, j+1, i]
                valid_weight += 0.5
            
            # Normalize by the actual number of valid ocean neighbors
            u_temp[k, j, i] = u_sum / valid_weight
            v_temp[k, j, i] = v_sum / valid_weight

@ti.kernel
def copy_uv_back_ti(u: ti.template(), v: ti.template(), u_temp: ti.template(), v_temp: ti.template()):
    for I in ti.grouped(u):
        u[I] = u_temp[I]
        v[I] = v_temp[I]
        
@ti.kernel
def advection_3d_tvd_ti(
    tracer: ti.template(), tracer_new: ti.template(),
    u: ti.template(), v: ti.template(), w: ti.template(),
    dz: ti.template(), dx: ti.template(), dy: ti.template(),
    dt: ti.f32, is_global: ti.i32
):
    nz, ny, nx = tracer.shape
    
    # 1. Copy over boundary values
    for k, j, i in tracer:
        tracer_new[k, j, i] = tracer[k, j, i]

    # 2. Compute Conservative 3D TVD advection for the inner domain
    for k, j, i in ti.ndrange((1, nz - 1), (1, ny - 1), nx):
        if is_global == 0 and (i == 0 or i == nx - 1):
            continue
            
        if dz[k, j, i] > 1e-6:
            c = tracer[k, j, i]
            
            # ==========================================
            # X-AXIS (Zonal) Interface Velocities
            # ==========================================
            i_w = i - 1 if (is_global == 1 or i > 0) else 0
            i_e = i + 1 if (is_global == 1 or i < nx - 1) else nx - 1
            if is_global == 1:
                if i_w < 0: i_w += nx
                if i_e >= nx: i_e -= nx

            # STRICT WALL BLOCKING: If adjacent cell is land, flux is 0.0
            u_w = 0.0 if dz[k, j, i_w] < 1e-6 else 0.5 * (u[k, j, i_w] + u[k, j, i])
            u_e = 0.0 if dz[k, j, i_e] < 1e-6 else 0.5 * (u[k, j, i] + u[k, j, i_e])
            
            # --- West Face Flux ---
            i_up_w = i_w if u_w > 0.0 else i
            i_dn_w = i if u_w > 0.0 else i_w
            i_up_up_w = i_w - 1 if u_w > 0.0 else i + 1
            if is_global == 1:
                if i_up_up_w < 0: i_up_up_w += nx
                elif i_up_up_w >= nx: i_up_up_w -= nx
            else:
                # CLAMP ALL INDICES TO PREVENT OUT-OF-BOUNDS AT OPEN EDGES
                i_up_w = ti.max(0, ti.min(nx - 1, i_up_w))
                i_dn_w = ti.max(0, ti.min(nx - 1, i_dn_w))
                i_up_up_w = ti.max(0, ti.min(nx - 1, i_up_up_w))

            c_up_w = tracer[k, j, i_up_w]
            c_dn_w = tracer[k, j, i_dn_w]
            c_up_up_w = tracer[k, j, i_up_up_w]
            
            denom_w = c_dn_w - c_up_w
            r_w = 0.0
            if ti.abs(denom_w) > 1e-8:
                r_w = (c_up_w - c_up_up_w) / denom_w
                
            cfl_w = ti.abs(u_w) * dt / dx[j, i]
            
            # EXPLICIT 1ST-ORDER UPWIND FALLBACK AT WESTERN BOUNDARY
            flux_w = 0.0
            if is_global == 0 and (i <= 1 or i >= nx - 1):
                flux_w = u_w * c_up_w
            else:
                flux_w = u_w * c_up_w + 0.5 * ti.abs(u_w) * (1.0 - cfl_w) * van_leer_limiter(r_w) * denom_w

            # --- East Face Flux ---
            i_up_e = i if u_e > 0.0 else i_e
            i_dn_e = i_e if u_e > 0.0 else i
            i_up_up_e = i - 1 if u_e > 0.0 else i_e + 1
            if is_global == 1:
                if i_up_up_e < 0: i_up_up_e += nx
                elif i_up_up_e >= nx: i_up_up_e -= nx
                
                if i_up_e >= nx: i_up_e -= nx
                if i_dn_e >= nx: i_dn_e -= nx
            else:
                # CLAMP ALL INDICES TO PREVENT OUT-OF-BOUNDS AT OPEN EDGES
                i_up_e = ti.max(0, ti.min(nx - 1, i_up_e))
                i_dn_e = ti.max(0, ti.min(nx - 1, i_dn_e))
                i_up_up_e = ti.max(0, ti.min(nx - 1, i_up_up_e))

            c_up_e = tracer[k, j, i_up_e]
            c_dn_e = tracer[k, j, i_dn_e]
            c_up_up_e = tracer[k, j, i_up_up_e]
            
            denom_e = c_dn_e - c_up_e
            r_e = 0.0
            if ti.abs(denom_e) > 1e-8:
                r_e = (c_up_e - c_up_up_e) / denom_e
                
            cfl_e = ti.abs(u_e) * dt / dx[j, i]
            
            # EXPLICIT 1ST-ORDER UPWIND FALLBACK AT EASTERN BOUNDARY
            flux_e = 0.0
            if is_global == 0 and (i <= 0 or i >= nx - 2):
                flux_e = u_e * c_up_e
            else:
                flux_e = u_e * c_up_e + 0.5 * ti.abs(u_e) * (1.0 - cfl_e) * van_leer_limiter(r_e) * denom_e

            term_x = (flux_e - flux_w) / dx[j, i]

            # ==========================================
            # Y-AXIS (Meridional) Interface Velocities
            # ==========================================
            j_s = ti.max(0, j - 1)
            j_n = ti.min(ny - 1, j + 1)
            
            # STRICT WALL BLOCKING: If adjacent cell is land, flux is 0.0
            v_s = 0.0 if dz[k, j_s, i] < 1e-6 else 0.5 * (v[k, j_s, i] + v[k, j, i])
            v_n = 0.0 if dz[k, j_n, i] < 1e-6 else 0.5 * (v[k, j, i] + v[k, j_n, i])
            
            # --- South Face Flux ---
            j_up_s = j_s if v_s > 0.0 else j
            j_dn_s = j if v_s > 0.0 else j_s
            j_up_up_s = ti.max(0, ti.min(ny - 1, j_s - 1 if v_s > 0.0 else j + 1))
            
            # CLAMP ALL INDICES TO PREVENT OUT-OF-BOUNDS AT OPEN EDGES
            if is_global == 0:
                j_up_s = ti.max(0, ti.min(ny - 1, j_up_s))
                j_dn_s = ti.max(0, ti.min(ny - 1, j_dn_s))
                j_up_up_s = ti.max(0, ti.min(ny - 1, j_up_up_s))
            
            c_up_s = tracer[k, j_up_s, i]
            c_dn_s = tracer[k, j_dn_s, i]
            c_up_up_s = tracer[k, j_up_up_s, i]
            
            denom_s = c_dn_s - c_up_s
            r_s = 0.0
            if ti.abs(denom_s) > 1e-8:
                r_s = (c_up_s - c_up_up_s) / denom_s
                
            cfl_s = ti.abs(v_s) * dt / dy[j, i]
            
            # EXPLICIT 1ST-ORDER UPWIND FALLBACK AT SOUTHERN BOUNDARY
            flux_s = 0.0
            if is_global == 0 and (j <= 1 or j >= ny - 1):
                flux_s = v_s * c_up_s
            else:
                flux_s = v_s * c_up_s + 0.5 * ti.abs(v_s) * (1.0 - cfl_s) * van_leer_limiter(r_s) * denom_s

            # --- North Face Flux ---
            j_up_n = j if v_n > 0.0 else j_n
            j_dn_n = j_n if v_n > 0.0 else j
            j_up_up_n = ti.max(0, ti.min(ny - 1, j - 1 if v_n > 0.0 else j_n + 1))
            
            # CLAMP ALL INDICES TO PREVENT OUT-OF-BOUNDS AT OPEN EDGES
            if is_global == 0:
                j_up_n = ti.max(0, ti.min(ny - 1, j_up_n))
                j_dn_n = ti.max(0, ti.min(ny - 1, j_dn_n))
                j_up_up_n = ti.max(0, ti.min(ny - 1, j_up_up_n))
            
            c_up_n = tracer[k, j_up_n, i]
            c_dn_n = tracer[k, j_dn_n, i]
            c_up_up_n = tracer[k, j_up_up_n, i]
            
            denom_n = c_dn_n - c_up_n
            r_n = 0.0
            if ti.abs(denom_n) > 1e-8:
                r_n = (c_up_n - c_up_up_n) / denom_n
                
            cfl_n = ti.abs(v_n) * dt / dy[j, i]
            
            # EXPLICIT 1ST-ORDER UPWIND FALLBACK AT NORTHERN BOUNDARY
            flux_n = 0.0
            if is_global == 0 and (j <= 0 or j >= ny - 2):
                flux_n = v_n * c_up_n
            else:
                flux_n = v_n * c_up_n + 0.5 * ti.abs(v_n) * (1.0 - cfl_n) * van_leer_limiter(r_n) * denom_n

            term_y = (flux_n - flux_s) / dy[j, i]

            # ==========================================
            # Z-AXIS (Vertical) Conservative TVD Fluxes
            # ==========================================
            k_t = k - 1 
            k_b = k + 1 
            
            w_t = 0.5 * (w[k_t, j, i] + w[k, j, i])
            w_b = 0.5 * (w[k, j, i] + w[k_b, j, i]) if k < nz - 1 else w[k, j, i]
            
            # --- Top Face Flux ---
            k_up_t = k_t + 1 if w_t > 0.0 else k_t - 1
            k_dn_t = k_t - 1 if w_t > 0.0 else k_t + 1
            k_up_up_t = k_t + 2 if w_t > 0.0 else k_t - 2
            
            k_up_t = ti.max(0, ti.min(nz - 1, k_up_t))
            k_dn_t = ti.max(0, ti.min(nz - 1, k_dn_t))
            k_up_up_t = ti.max(0, ti.min(nz - 1, k_up_up_t))
            
            c_up_t = tracer[k_up_t, j, i]
            c_dn_t = tracer[k_dn_t, j, i]
            c_up_up_t = tracer[k_up_up_t, j, i]
            
            denom_t = c_dn_t - c_up_t
            r_t = 0.0
            if ti.abs(denom_t) > 1e-8:
                r_t = (c_up_t - c_up_up_t) / denom_t
                
            cfl_t = ti.abs(w_t) * dt / dz[k, j, i]
            flux_t = w_t * c_up_t + 0.5 * ti.abs(w_t) * (1.0 - cfl_t) * minmod_limiter(r_t) * denom_t

            # --- Bottom Face Flux ---
            k_up_b = k_b + 1 if w_b > 0.0 else k_b - 1
            k_dn_b = k_b - 1 if w_b > 0.0 else k_b + 1
            k_up_up_b = k_b + 2 if w_b > 0.0 else k_b - 2
            
            k_up_b = ti.max(0, ti.min(nz - 1, k_up_b))
            k_dn_b = ti.max(0, ti.min(nz - 1, k_dn_b))
            k_up_up_b = ti.max(0, ti.min(nz - 1, k_up_up_b))
            
            c_up_b = tracer[k_up_b, j, i]
            c_dn_b = tracer[k_dn_b, j, i]
            c_up_up_b = tracer[k_up_up_b, j, i]
            
            denom_b = c_dn_b - c_up_b
            r_b = 0.0
            if ti.abs(denom_b) > 1e-8:
                r_b = (c_up_b - c_up_up_b) / denom_b
                
            cfl_b = ti.abs(w_b) * dt / dz[k, j, i]
            flux_b = w_b * c_up_b + 0.5 * ti.abs(w_b) * (1.0 - cfl_b) * minmod_limiter(r_b) * denom_b

            # Net Divergence Z
            # (Flux passing upward through top face minus flux passing upward through bottom face)
            term_z = (flux_t - flux_b) / dz[k, j, i]  
            
            # ==========================================
            # DIVERGENCE CORRECTION (Advective Form)
            # ==========================================
            # 1. Calculate the raw numerical volume divergence of the 3D cell
            div_u = (u_e - u_w) / dx[j, i]
            div_v = (v_n - v_s) / dy[j, i]
            div_w = (w_t - w_b) / dz[k, j, i]
            total_div = div_u + div_v + div_w
            
            # 2. Subtract the artificial concentration caused by water compression (- c * total_div)
            tracer_new[k, j, i] = ti.max(0.0, c - dt * (term_x + term_y + term_z - c * total_div))
            
@ti.kernel
def diffusion_implicit_vertical_ti(
    tracer: ti.template(), kz: ti.template(), dz: ti.template(),
    cp: ti.template(), dp: ti.template(), dt: ti.f32
):
    nz, ny, nx = tracer.shape
    
    # Massively parallelize across the horizontal domain
    for j, i in ti.ndrange(ny, nx):
        if dz[0, j, i] > 1e-6:
            
            # 1. Find the actual ocean bottom for this specific column
            k_bot = 0
            for k in range(nz):
                if dz[k, j, i] > 1e-6:
                    k_bot = k
                    
            if k_bot > 0:
                # ==========================================
                # FORWARD SWEEP (Top to Bottom)
                # ==========================================
                
                # --- Surface Cell (k = 0) ---
                dz_eff_bot = 0.5 * (dz[0, j, i] + dz[1, j, i])
                k_eff_bot = 0.5 * (kz[0, j, i] + kz[1, j, i])
                
                # A_0 (upper diagonal) is 0 because there is no flux through the ocean surface
                C_0 = -(dt / dz[0, j, i]) * (k_eff_bot / dz_eff_bot)
                B_0 = 1.0 - C_0
                
                cp[0, j, i] = C_0 / B_0
                dp[0, j, i] = tracer[0, j, i] / B_0
                
                # --- Interior Cells (k = 1 to k_bot - 1) ---
                for k in range(1, k_bot):
                    dz_eff_top = 0.5 * (dz[k-1, j, i] + dz[k, j, i])
                    k_eff_top = 0.5 * (kz[k-1, j, i] + kz[k, j, i])
                    
                    dz_eff_bot = 0.5 * (dz[k, j, i] + dz[k+1, j, i])
                    k_eff_bot = 0.5 * (kz[k, j, i] + kz[k+1, j, i])
                    
                    A_k = -(dt / dz[k, j, i]) * (k_eff_top / dz_eff_top)
                    C_k = -(dt / dz[k, j, i]) * (k_eff_bot / dz_eff_bot)
                    B_k = 1.0 - A_k - C_k
                    
                    denom = B_k - A_k * cp[k-1, j, i]
                    cp[k, j, i] = C_k / denom
                    dp[k, j, i] = (tracer[k, j, i] - A_k * dp[k-1, j, i]) / denom
                    
                # --- Bottom Cell (k = k_bot) ---
                dz_eff_top = 0.5 * (dz[k_bot-1, j, i] + dz[k_bot, j, i])
                k_eff_top = 0.5 * (kz[k_bot-1, j, i] + kz[k_bot, j, i])
                
                A_k = -(dt / dz[k_bot, j, i]) * (k_eff_top / dz_eff_top)
                # C_k is 0 because there is no flux into the sea floor
                B_k = 1.0 - A_k
                
                denom = B_k - A_k * cp[k_bot-1, j, i]
                dp[k_bot, j, i] = (tracer[k_bot, j, i] - A_k * dp[k_bot-1, j, i]) / denom
                
                # ==========================================
                # BACKWARD SUBSTITUTION (Bottom to Top)
                # ==========================================
                
                # Set the bottom tracer value
                tracer[k_bot, j, i] = dp[k_bot, j, i]
                
                # Work our way back up to the surface, solving the equations in-place
                for step in range(1, k_bot + 1):
                    k = k_bot - step
                    tracer[k, j, i] = dp[k, j, i] - cp[k, j, i] * tracer[k+1, j, i]
            
@ti.kernel
def calculate_w_rigid_lid_ti(
    u: ti.template(), v: ti.template(), dz: ti.template(),
    dx: ti.template(), dy: ti.template(), w: ti.template()
):
    nz, ny, nx = u.shape
    
    # Clear W array
    for k, j, i in w:
        w[k, j, i] = 0.0
        
    for j, i in ti.ndrange((1, ny-1), (1, nx-1)):
        k_bottom = 0
        for k in range(nz):
            if dz[k, j, i] > 1e-6:
                k_bottom += 1
                
        if k_bottom > 0:
            w_accum = 0.0
            for k in range(k_bottom):
                # STRICT WALL BLOCKING: 0.0 flux if neighbor is land
                u_w = 0.5 * (u[k, j, i-1] + u[k, j, i]) if dz[k, j, i-1] > 1e-6 else 0.0
                u_e = 0.5 * (u[k, j, i] + u[k, j, i+1]) if dz[k, j, i+1] > 1e-6 else 0.0
                v_s = 0.5 * (v[k, j-1, i] + v[k, j, i]) if dz[k, j-1, i] > 1e-6 else 0.0
                v_n = 0.5 * (v[k, j, i] + v[k, j+1, i]) if dz[k, j+1, i] > 1e-6 else 0.0
                
                # Calculate horizontal divergence
                div_h = ((u_e - u_w) / dx[j, i]) + ((v_n - v_s) / dy[j, i])
                
                # Accumulate vertical velocity (integrating from surface down)
                w_accum += div_h * dz[k, j, i]
                w[k+1, j, i] = w_accum
                
            # Rigid lid correction: ensure W at the ocean floor is exactly 0.0
            bottom_error = w[k_bottom, j, i]
            for k in range(1, k_bottom + 1):
                w[k, j, i] -= bottom_error * float(k) / float(k_bottom)

@ti.kernel
def calculate_mld_ti(rho: ti.template(), dz: ti.template(), mld: ti.template()):
    nz, ny, nx = rho.shape
    for j, i in ti.ndrange(ny, nx):
        if dz[0, j, i] < 1e-6 or ti.math.isnan(rho[0, j, i]):
            mld[j, i] = ti.math.nan
            continue
            
        depth = 0.0
        k_ref = 0
        for k in range(nz):
            depth += dz[k, j, i]
            if depth >= 10.0:
                k_ref = k
                break
                
        rho_ref = rho[k_ref, j, i]
        found = 0
        current_depth = 0.0
        for k in range(nz):
            if k >= k_ref and (rho[k, j, i] - rho_ref) > 0.03 and found == 0:
                mld[j, i] = current_depth
                found = 1
            current_depth += dz[k, j, i]
            
        if found == 0:
            mld[j, i] = current_depth

@ti.kernel
def calculate_full_kz_ti(
    mld: ti.template(), rho: ti.template(), dz: ti.template(), kz: ti.template(),
    k_mld_max: ti.f32, k_bg_min: ti.f32, k_deep_max: ti.f32, k_boundary: ti.f32
):
    nz, ny, nx = rho.shape
    for j, i in ti.ndrange(ny, nx):
        if dz[0, j, i] < 1e-6:
            for k in range(nz):
                kz[k, j, i] = ti.math.nan
            continue
            
        current_depth = 0.0
        for k in range(nz):
            current_depth += dz[k, j, i]
            
            is_bottom = (k == nz-1) or (dz[k+1, j, i] < 1e-6)
            is_coast = False
            if i > 0 and dz[k, j, i-1] < 1e-6: is_coast = True
            if i < nx-1 and dz[k, j, i+1] < 1e-6: is_coast = True
            if j > 0 and dz[k, j-1, i] < 1e-6: is_coast = True
            if j < ny-1 and dz[k, j+1, i] < 1e-6: is_coast = True
            
            local_bg = k_boundary if (is_bottom or is_coast) else k_bg_min
            local_ceil = k_boundary if (is_bottom or is_coast) else k_deep_max
            
            if current_depth <= mld[j, i]:
                sigma = current_depth / mld[j, i]
                kz[k, j, i] = local_bg + (k_mld_max * 6.75 * sigma * (1.0 - sigma)**2)
            else:
                N2 = 1e-7
                if k < nz - 1 and dz[k+1, j, i] > 1e-6:
                    drho = rho[k+1, j, i] - rho[k, j, i]
                    dz_eff = 0.5 * (dz[k, j, i] + dz[k+1, j, i])
                    N2 = ti.max((9.81 / 1035.0) * (drho / dz_eff), 1e-7)
                k_deep = 1e-11 / ti.sqrt(N2)
                kz[k, j, i] = ti.min(ti.max(k_deep, local_bg), local_ceil)

@ti.kernel
def calc_o2_flux_ti(
    o2_surf: ti.template(), o2_sat: ti.template(), temp: ti.template(),
    wind: ti.template(), ice: ti.template(), dz: ti.template(), dt: ti.f32
):
    ny, nx = o2_surf.shape
    for j, i in ti.ndrange(ny, nx):
        if dz[j, i] > 1e-6:
            ts = temp[j, i]
            sc_o2 = ti.max(1.0, 1920.4 - 135.6*ts + 5.2122*(ts**2) - 0.10939*(ts**3) + 0.00093777*(ts**4))
            kw_m_s = (0.251 * (wind[j, i]**2) * ti.math.pow(sc_o2 / 660.0, -0.5)) / 360000.0
            open_water = ti.max(0.0, 1.0 - ice[j, i])
            flux = kw_m_s * (o2_sat[j, i] - o2_surf[j, i]) * open_water
            o2_surf[j, i] += (flux / dz[j, i]) * dt