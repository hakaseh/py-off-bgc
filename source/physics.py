import numpy as np
import taichi as ti
import math

# ==========================================
# 1. SETUP FUNCTIONS (Pure NumPy - Runs Once)
# ==========================================

def generate_static_grid(ds_template):
    R_EARTH = 6371000.0
    dlon_vals = np.abs(np.diff(ds_template['lon'].values))
    dlon_vals = np.append(dlon_vals, dlon_vals[-1])
    dlat_vals = np.abs(np.diff(ds_template['lat'].values))
    dlat_vals = np.append(dlat_vals, dlat_vals[-1])
    
    dlon_rad = np.radians(dlon_vals)[:, None] * np.ones((len(dlat_vals), len(dlon_vals)))
    dlat_rad = np.radians(dlat_vals)[:, None] * np.ones((len(dlat_vals), len(dlon_vals)))
    lat_rad = np.radians(ds_template['lat'].values)[:, None] * np.ones((len(dlat_vals), len(dlon_vals)))
    
    area = (R_EARTH**2) * dlat_rad * dlon_rad * np.cos(lat_rad)
    
    import xarray as xr
    ds_grid = xr.Dataset(
        data_vars={'area': (['lat', 'lon'], area)},
        coords=ds_template.coords
    )
    
    if 'depth' in ds_template.coords:
        ddep_vals = np.abs(np.diff(ds_template['depth'].values))
        ddep_vals = np.append(ddep_vals, ddep_vals[-1])
        ds_grid['dz'] = (['depth'], ddep_vals)
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

@ti.kernel
def advection_neumann_ti(
    tracer: ti.template(), tracer_new: ti.template(),
    u: ti.template(), v: ti.template(), w: ti.template(),
    dz: ti.template(), dx: ti.template(), dy: ti.template(),
    dt: ti.f32, is_global: ti.i32
):
    nz, ny, nx = tracer.shape
    
    # 1. Copy over boundary values natively
    for k, j, i in tracer:
        tracer_new[k, j, i] = tracer[k, j, i]

    # 2. Compute advection exclusively for the inner domain
    # ti.ndrange natively loops across the GPU.
    for k, j, i in ti.ndrange(nz, (1, ny - 1), nx):
        # Skip boundaries if regional
        if is_global == 0 and (i == 0 or i == nx - 1):
            continue
            
        if dz[k, j, i] > 1e-6:
            c = tracer[k, j, i]
            
            # Horizontal (Upwind)
            i_up = i - 1 if u[k, j, i] > 0.0 else i + 1
            j_up = j - 1 if v[k, j, i] > 0.0 else j + 1
            
            if is_global == 1:
                if i_up < 0: i_up = nx - 1
                elif i_up >= nx: i_up = 0
                
            c_up_x = tracer[k, j, i_up] if dz[k, j, i_up] > 1e-6 else c
            c_up_y = tracer[k, j_up, i] if dz[k, j_up, i] > 1e-6 else c
            
            term_x = ti.abs(u[k, j, i]) * (c - c_up_x) / dx[j, i]
            term_y = ti.abs(v[k, j, i]) * (c - c_up_y) / dy[j, i]
            
            # Vertical
            # Guard against out-of-bounds k+1 access at the ocean floor
            w_val = 0.0
            if k < nz - 1:
                w_val = 0.5 * (w[k, j, i] + w[k+1, j, i])
            else:
                w_val = 0.5 * w[k, j, i]
                
            c_below = tracer[k+1, j, i] if (k < nz - 1 and dz[k+1, j, i] > 1e-6) else c
            c_above = tracer[k-1, j, i] if (k > 0 and dz[k-1, j, i] > 1e-6) else c
            
            # FIX: Pre-declare term_z outside the if-block so it survives!
            term_z = 0.0
            if w_val > 0.0:
                term_z = w_val * (c - c_below) / dz[k, j, i]
            else:
                term_z = w_val * (c_above - c) / dz[k, j, i]
                
            tracer_new[k, j, i] = ti.max(0.0, c - dt * (term_x + term_y + term_z))
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
                u_w = 0.5 * (u[k, j, i-1] + u[k, j, i]) if dz[k, j, i-1] > 1e-6 else 0.0
                u_e = 0.5 * (u[k, j, i] + u[k, j, i+1]) if dz[k, j, i+1] > 1e-6 else 0.0
                v_s = 0.5 * (v[k, j-1, i] + v[k, j, i]) if dz[k, j-1, i] > 1e-6 else 0.0
                v_n = 0.5 * (v[k, j, i] + v[k, j+1, i]) if dz[k, j+1, i] > 1e-6 else 0.0
                
                div_h = ((u_e - u_w) / dx[j, i]) + ((v_n - v_s) / dy[j, i])
                w_accum += div_h * dz[k, j, i]
                w[k+1, j, i] = w_accum
                
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
def diffusion_robust_ti(tracer: ti.template(), kz: ti.template(), dz: ti.template(), dt: ti.f32):
    nz, ny, nx = tracer.shape
    for j, i in ti.ndrange(ny, nx):
        if dz[0, j, i] > 1e-6:
            for k in range(nz):
                if dz[k, j, i] < 1e-6: continue
                
                flux_top, flux_bot = 0.0, 0.0
                if k > 0:
                    delta_z_top = 0.5 * (dz[k-1, j, i] + dz[k, j, i])
                    k_eff_top = ti.min(0.5 * (kz[k-1, j, i] + kz[k, j, i]), 0.45 * (delta_z_top**2) / dt)
                    flux_top = k_eff_top * (tracer[k, j, i] - tracer[k-1, j, i]) / delta_z_top if delta_z_top > 1e-6 else 0.0
                
                if k < nz - 1:
                    delta_z_bot = 0.5 * (dz[k, j, i] + dz[k+1, j, i])
                    k_eff_bot = ti.min(0.5 * (kz[k, j, i] + kz[k+1, j, i]), 0.45 * (delta_z_bot**2) / dt)
                    flux_bot = k_eff_bot * (tracer[k+1, j, i] - tracer[k, j, i]) / delta_z_bot if delta_z_bot > 1e-6 else 0.0
                
                tracer[k, j, i] += dt * (flux_bot - flux_top) / dz[k, j, i]

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