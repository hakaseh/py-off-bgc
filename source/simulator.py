import numpy as np
import xarray as xr
import pandas as pd
import os
import gsw
from . import physics
from .bgc_models.utils import GLODAP_MAP

class OfflineSimulator:
    def __init__(self, 
                 da_t, 
                 bgc_model, 
                 model_name, 
                 exp_name, 
                 extra_name, 
                 dt_phys, 
                 ds_clim = None,
                 mixing_method = "diffusion", 
                 mld_threshold = 0.03,
                 sponge_width = 5, 
                 tau_lateral=432000.0, 
                 tau_bottom=5184000.0,
                 is_global = False
                ):
        
        self.bgc_model = bgc_model
        self.model_name = model_name
        self.exp_name = exp_name
        self.extra_name = extra_name
        self.dt_phys = dt_phys
        self.steps_per_day = int(86400 / self.dt_phys)        
        self.ds_clim = ds_clim
        self.mixing_method = mixing_method 
        self.mld_threshold = mld_threshold
        self.is_global = is_global
        
        self.setup_grid(da_t, sponge_width, tau_lateral, tau_bottom)
        self.setup_io(da_t)
        self.setup_restoring()
        
    def setup_restoring(self):
        """Dynamically load restoring targets from the prepared climatology dataset."""
        self.restoring_data = {}
        if self.ds_clim is None:
            print("  [Warning] No climatology dataset provided for restoring.")
            return
            
        print("Loading Restoring Targets (Sponge)...")
        for model_var in self.bgc_model.names:
            clim_var = GLODAP_MAP.get(model_var, model_var)
            if clim_var in self.ds_clim:
                print(f"  -> Found restoring field for: {model_var}")
                self.restoring_data[model_var] = np.nan_to_num(self.ds_clim[clim_var].values)

    def setup_grid(self, da_ref, sponge_width, tau_lateral, tau_bottom):
            print("Initializing Grid...")
            
            # 1. Metrics & Coordinates (Save 2D lons/lats for gsw density)
            if da_ref['lat'].ndim == 2:
                self.dx, self.dy = physics.calculate_metrics_curvilinear(da_ref['lon'].values, da_ref['lat'].values)
                self.lon_2d = da_ref['lon'].values
                self.lat_2d = da_ref['lat'].values
            else:
                self.dx, self.dy = physics.calculate_grid_metrics(da_ref)
                self.lon_2d, self.lat_2d = np.meshgrid(da_ref['lon'].values, da_ref['lat'].values)
                
            # 2. Vertical Grid
            dz_1d = physics.calculate_dz_from_centers(da_ref['depth'].values)
            nz, ny, nx = da_ref.shape[1:] 
            if len(da_ref.shape) == 3: nz, ny, nx = da_ref.shape
                
            self.dz_3d = np.tile(dz_1d[:, None, None], (1, ny, nx))
            
            # Save 3D pressure (approx depth) for gsw calculations
            self.p_3d = da_ref['depth'].values[:, None, None]
            
            # 3. Mask
            ref_val = da_ref.isel(time=0).values if 'time' in da_ref.dims else da_ref.values
            self.water_mask = ~np.isnan(ref_val)
            self.dz_static = np.where(self.water_mask, self.dz_3d, 0.0)
            
            # 4. Restoring Weights (Sponge)
            print("Generating Restoring Map...")
            self.nudge_map = physics.create_restoring_weights(
                self.water_mask, sponge_width, tau_lateral, tau_bottom, self.dt_phys, self.is_global
            )

    def setup_io(self, da_ref):
        self.ds_template = da_ref.isel(time=0).drop_vars('time')
        self.out_dir = f"output/{self.model_name}_{self.exp_name}_{self.extra_name}"
        os.makedirs(self.out_dir, exist_ok=True)
        
    def run(self, da_u, da_v, da_k, da_t, da_s, da_sw): 
        nt = da_t['time'].size
        print(f"Starting Simulation ({nt} days)...")
        
        for day in range(nt):
            print(f"  Day {day+1} / {nt}")
            
            # 1. Load Forcing
            u = np.nan_to_num(da_u.isel(time=day).values)
            v = np.nan_to_num(da_v.isel(time=day).values)
            k = np.nan_to_num(da_k.isel(time=day).values)
            t = np.nan_to_num(da_t.isel(time=day).values)
            s = np.nan_to_num(da_s.isel(time=day).values)
            sw = np.nan_to_num(da_sw.isel(time=day).values)
            
            # 2. Calculate Density and MLD
            SA = gsw.SA_from_SP(s, self.p_3d, self.lon_2d, self.lat_2d)
            CT = gsw.CT_from_pt(SA, t)
            rho_3d = gsw.sigma0(SA, CT)
            self.mld_2d = physics.calculate_mld(rho_3d, self.dz_static, self.mld_threshold)
            
            # 2. Calc W
            u[~self.water_mask] = 0.0
            v[~self.water_mask] = 0.0
            w = physics.calculate_w(u, v, self.dz_static, self.dx, self.dy)  
            
            # --- SUB-STEPPED LOOP (Physics + Biology + Restoring) ---
            for step in range(self.steps_per_day):
                
                # A. PHYSICS
                for name, tr_data in self.bgc_model.tracers.items():
                    # Advection
                    tr_adv = physics.advection_neumann(tr_data, u, v, w, self.dz_static, self.dt_phys, self.dx, self.dy, is_global=self.is_global)
                    
                    # Mixing
                    if self.mixing_method == "diffusion":
                        tr_mix = physics.diffusion_robust(tr_adv, k, self.dz_static, self.dt_phys)
                    elif self.mixing_method == "convective":
                        tr_mix = physics.mixing_convective(tr_adv, rho_3d, self.dz_static, self.mld_threshold)
                        
                    self.bgc_model.tracers[name][:] = np.maximum(tr_mix, 0.0)

                # B. RESTORING (Sponge)
                # Using the original nudge_map which is scaled for dt_phys
                for var_name, clim_data in self.restoring_data.items():
                    diff = clim_data - self.bgc_model.tracers[var_name]
                    self.bgc_model.tracers[var_name] += diff * self.nudge_map

                # C. BIOLOGY & SINKING
                # Passing the small dt_phys to prevent non-linear overshoots
                par_3d = self.bgc_model.biology_step(t, sw, self.dz_static, self.dt_phys)
                self.bgc_model.sinking_step(self.dz_static, self.dt_phys)
                
                # Final safety clamp
                for name, tr_data in self.bgc_model.tracers.items():
                    self.bgc_model.tracers[name][:] = np.maximum(tr_data, 0.0)
                                    
            # ... Save Output ...              
            self.save_day(day, da_t.isel(time=day).time.values, par_3d)

    def save_day(self, day, current_time, par_3d):
        # 1. Gather 3D Data
        data_map = {name: arr for name, arr in self.bgc_model.tracers.items()}
        data_map['PAR'] = par_3d
        
        ds_out = xr.Dataset(coords=self.ds_template.coords)
        
        # 2. Save 3D Variables
        for var, arr in data_map.items():
            ds_out[var] = (self.ds_template.dims, np.where(self.water_mask, arr, np.nan))
            ds_out[var].attrs = {'units': 'mmol/m3'}
            
        # 3. Save 2D Variables (MLD)
        # Find the 2D dimension names (e.g., ('lat', 'lon')) by ignoring 'depth'
        dims_2d = [dim for dim in self.ds_template.dims if dim != 'depth']
        mask_2d = self.water_mask[0, :, :] # Surface mask
        
        ds_out['MLD'] = (dims_2d, np.where(mask_2d, self.mld_2d, np.nan))
        ds_out['MLD'].attrs = {'units': 'm', 'long_name': 'Mixed Layer Depth'}
            
        # 4. Finalize and Write to Disk
        ds_out = ds_out.expand_dims(time=[current_time])
        
        t_str = pd.to_datetime(current_time).strftime('%Y%m%d')
        fname = f"{self.out_dir}/output_{self.model_name}_{self.exp_name}_{t_str}.nc"
        
        if os.path.exists(fname):
            try:
                os.remove(fname)  
            except PermissionError:
                print(f"  [Error] Cannot overwrite {fname}.")
                return 
                
        comp = dict(zlib=True, complevel=5)
        enc = {v: comp for v in ds_out.data_vars}
        ds_out.to_netcdf(fname, encoding=enc)
        
        print(f"    -> Saved {fname}")
        if 'nitrate' in data_map:
            print(f"    -> Nitrate Mean, Min, Max: {np.nanmean(data_map['nitrate']):.1f}, {np.nanmin(data_map['nitrate']):.1f}, {np.nanmax(data_map['nitrate']):.1f}")