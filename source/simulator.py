import numpy as np
import xarray as xr
import pandas as pd
import os
import gsw
from . import physics
from . import bgc_models
from .bgc_models.utils import GLODAP_MAP

class OfflineSimulator:
    def __init__(self, 
                 bgc_model_choice, 
                 exp_name, 
                 dt_phys, 
                 bgc_params=None,
                 mld_threshold=0.03,
                 sponge_width=5, 
                 tau_lateral=432000.0, 
                 tau_bottom=5184000.0,
                 is_global=False):
        
        # 1. Store settings
        self.bgc_model_choice = bgc_model_choice
        self.exp_name = exp_name
        self.dt_phys = dt_phys
        self.steps_per_day = int(86400 / self.dt_phys)
        self.bgc_params = bgc_params
        self.mld_threshold = mld_threshold
        self.sponge_width = sponge_width
        self.tau_lateral = tau_lateral
        self.tau_bottom = tau_bottom
        self.is_global = is_global
        
        # These will be set during prepare_forcing()
        self.bgc_model = None
        self.ds_clim = None
        self.mixing_method = None

    def prepare_forcing(self, lat_range, lon_range, depth_range, time_range, 
                        ds_t, ds_s, ds_u, ds_v, ds_sw, ds_wind, 
                        ds_ice=None, ds_k=None, ds_clim=None, ds_restart=None):
        """
        Subsets all datasets, handles missing inputs, and initializes the BGC model.
        """
        print("Preparing and subsetting forcing datasets...")
        
        # 1. Subset all core physical data
        self.ds_t = ds_t.sel(time=time_range, depth=depth_range, lat=lat_range, lon=lon_range)
        self.ds_s = ds_s.sel(time=time_range, depth=depth_range, lat=lat_range, lon=lon_range)
        self.ds_u = ds_u.sel(time=time_range, depth=depth_range, lat=lat_range, lon=lon_range)
        self.ds_v = ds_v.sel(time=time_range, depth=depth_range, lat=lat_range, lon=lon_range)
        self.ds_sw = ds_sw.sel(time=time_range, lat=lat_range, lon=lon_range)

        # 2. Handle optional/missing data using the subsetted grid size
        if ds_k is None:
            print("  -> No Kz file provided. Using dummy array and 'convective' mixing.")
            self.ds_k = xr.zeros_like(self.ds_t) 
            self.mixing_method = "convective"
        else:
            self.ds_k = ds_k.sel(time=time_range, depth=depth_range, lat=lat_range, lon=lon_range)
            self.mixing_method = "diffusion"

        if ds_wind is None:
            print("  -> No win file provided. Setting all to ZERO (no gas exchange).")
            self.ds_wind = xr.zeros_like(self.ds_sw)
        else:
            self.ds_wind = ds_wind.sel(time=time_range, lat=lat_range, lon=lon_range)

        if ds_ice is None:
            print("  -> No Ice file provided. Setting all to ONE (no ice cover).")
            self.ds_ice = xr.ones_like(self.ds_sw)
        else:
            self.ds_ice = ds_ice.sel(time=time_range, lat=lat_range, lon=lon_range)

        # 3. Subset BGC specific files
        if ds_clim is not None:
            self.ds_clim = ds_clim.sel(depth=depth_range, lat=lat_range, lon=lon_range)
        if ds_restart is not None:
            ds_restart = ds_restart.sel(depth=depth_range, lat=lat_range, lon=lon_range)

        # 4. Execute standard setup sequence
        self.setup_grid()
        
        # 5. Initialize the BGC Model now that the grid shape (nz, ny, nx) is known!
        print(f"Initializing BGC Model: {self.bgc_model_choice}...")
        self.bgc_model = bgc_models.get_model(self.bgc_model_choice, self.nz, self.ny, self.nx, self.water_mask, params=self.bgc_params)
        self.bgc_model.initialize(ds_restart=ds_restart, ds_clim=self.ds_clim)
        if ds_restart is not None:
            ds_restart.close()

        self.setup_io()
        self.setup_restoring()

    def setup_grid(self):
        print("Initializing Grid...")
        # 1. Metrics & Coordinates
        if self.ds_t['lat'].ndim == 2:
            self.dx, self.dy = physics.calculate_metrics_curvilinear(self.ds_t['lon'].values, self.ds_t['lat'].values)
            self.lon_2d = self.ds_t['lon'].values
            self.lat_2d = self.ds_t['lat'].values
        else:
            self.dx, self.dy = physics.calculate_grid_metrics(self.ds_t)
            self.lon_2d, self.lat_2d = np.meshgrid(self.ds_t['lon'].values, self.ds_t['lat'].values)
            
        # 2. Vertical Grid & Dimensions
        self.nz, self.ny, self.nx = self.ds_t.depth.size, self.ds_t.lat.size, self.ds_t.lon.size
        dz_1d = physics.calculate_dz_from_centers(self.ds_t['depth'].values)
        self.dz_3d = np.tile(dz_1d[:, None, None], (1, self.ny, self.nx))
        self.p_3d = self.ds_t['depth'].values[:, None, None]
        
        # 3. Mask
        ref_val = self.ds_t.isel(time=0).values if 'time' in self.ds_t.dims else self.ds_t.values
        self.water_mask = ~np.isnan(ref_val)
        self.dz_static = np.where(self.water_mask, self.dz_3d, 0.0)
        
        # 4. Restoring Weights (Sponge)
        print("Generating Restoring Map...")
        self.nudge_map = physics.create_restoring_weights(
            self.water_mask, self.sponge_width, self.tau_lateral, self.tau_bottom, self.dt_phys, self.is_global
        )

    def setup_io(self):
        self.ds_template = self.ds_t.isel(time=0).drop_vars('time')
        self.out_dir = f"output/{self.bgc_model_choice}_{self.exp_name}"
        os.makedirs(self.out_dir, exist_ok=True)
        
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

    def run(self): 
        nt = self.ds_t['time'].size
        print(f"Starting Simulation ({nt} days)...")
        
        for day in range(nt):
            print(f"  Day {day+1} / {nt}")
            
            # 1. Load Forcing (Directly from internally stored, subsetted datasets)
            u = np.nan_to_num(self.ds_u.isel(time=day).values)
            v = np.nan_to_num(self.ds_v.isel(time=day).values)
            k = np.nan_to_num(self.ds_k.isel(time=day).values)
            t = np.nan_to_num(self.ds_t.isel(time=day).values)
            s = np.nan_to_num(self.ds_s.isel(time=day).values)
            sw = np.nan_to_num(self.ds_sw.isel(time=day).values)
            wind_surf = np.nan_to_num(self.ds_wind.isel(time=day).values) 
            ice_surf = np.nan_to_num(self.ds_ice.isel(time=day).values)            
            
            # 2. Calculate Density and MLD
            SA = gsw.SA_from_SP(s, self.p_3d, self.lon_2d, self.lat_2d)
            CT = gsw.CT_from_pt(SA, t)
            rho_3d = gsw.sigma0(SA, CT)
            self.mld_2d = physics.calculate_mld(rho_3d, self.dz_static, self.mld_threshold)
            
            # Calc W
            u[~self.water_mask] = 0.0
            v[~self.water_mask] = 0.0
            w = physics.calculate_w(u, v, self.dz_static, self.dx, self.dy)  

            # Calculate O2 Saturation ONCE per day
            if "oxygen" in self.bgc_model.tracers:
                salt_surf = s[0, :, :]
                temp_surf = t[0, :, :]
                rho_surf = rho_3d[0, :, :]
                o2_sat_umol_kg = gsw.O2sol_SP_pt(salt_surf, temp_surf)
                o2_sat_mmol_m3 = o2_sat_umol_kg * (rho_surf / 1000.0)
            
            # --- SUB-STEPPED LOOP (Physics + Biology + Restoring) ---
            for step in range(self.steps_per_day):
                
                # A. PHYSICS
                for name, tr_data in self.bgc_model.tracers.items():
                    tr_adv = physics.advection_neumann(tr_data, u, v, w, self.dz_static, self.dt_phys, self.dx, self.dy, is_global=self.is_global)
                    
                    if self.mixing_method == "diffusion":
                        tr_mix = physics.diffusion_robust(tr_adv, k, self.dz_static, self.dt_phys)
                    elif self.mixing_method == "convective":
                        tr_mix = physics.mixing_convective(tr_adv, rho_3d, self.dz_static, self.mld_threshold)
                        
                    # CLAMP #1: Immediately after physics to fix advection overshoots
                    self.bgc_model.tracers[name][:] = np.maximum(tr_mix, 0.0)

                # B. RESTORING
                for var_name, clim_data in self.restoring_data.items():
                    diff = clim_data - self.bgc_model.tracers[var_name]
                    self.bgc_model.tracers[var_name] += diff * self.nudge_map

                # C. BIOLOGY & SINKING
                par_3d = self.bgc_model.biology_step(t, sw, self.dz_static, self.dt_phys)
                self.bgc_model.sinking_step(self.dz_static, self.dt_phys)

                # CLAMP #2: Immediately after biology/sinking
                for name, tr_data in self.bgc_model.tracers.items():
                    self.bgc_model.tracers[name][:] = np.maximum(tr_data, 0.0)

                # D. AIR-SEA FLUX (OXYGEN)
                if "oxygen" in self.bgc_model.tracers:
                    o2_surf = self.bgc_model.tracers["oxygen"][0, :, :]
                    temp_surf = t[0, :, :]
                    dz_surf = self.dz_static[0, :, :]
                    
                    updated_o2_surf = physics.calc_o2_flux(
                        o2_surf, o2_sat_mmol_m3, temp_surf, wind_surf, 
                        ice_surf, dz_surf, self.dt_phys
                    )
                    self.bgc_model.tracers["oxygen"][0, :, :] = np.maximum(updated_o2_surf, 0.0)
                
            # Save Output              
            self.save_day(day, self.ds_t.isel(time=day).time.values, par_3d)

    def save_day(self, day, current_time, par_3d):
        # ... (Keep this exactly the same as your current save_day method!) ...
        data_map = {name: arr for name, arr in self.bgc_model.tracers.items()}
        data_map['PAR'] = par_3d
        ds_out = xr.Dataset(coords=self.ds_template.coords)
        for var, arr in data_map.items():
            ds_out[var] = (self.ds_template.dims, np.where(self.water_mask, arr, np.nan))
            ds_out[var].attrs = {'units': 'mmol/m3'}
            
        dims_2d = [dim for dim in self.ds_template.dims if dim != 'depth']
        mask_2d = self.water_mask[0, :, :] 
        ds_out['MLD'] = (dims_2d, np.where(mask_2d, self.mld_2d, np.nan))
        ds_out['MLD'].attrs = {'units': 'm', 'long_name': 'Mixed Layer Depth'}
            
        ds_out = ds_out.expand_dims(time=[current_time])
        t_str = pd.to_datetime(current_time).strftime('%Y%m%d')
        fname = f"{self.out_dir}/output_{self.bgc_model_choice}_{self.exp_name}_{t_str}.nc"
        
        if os.path.exists(fname):
            try: os.remove(fname)  
            except PermissionError:
                print(f"  [Error] Cannot overwrite {fname}. Skipping save.")
                return 
                
        comp = dict(zlib=True, complevel=5)
        enc = {v: comp for v in ds_out.data_vars}
        ds_out.to_netcdf(fname, encoding=enc)
        
        print(f"    -> Saved {fname}")
        if 'nitrate' in data_map:
            print(f"    -> Nitrate Mean, Min, Max: {np.nanmean(data_map['nitrate']):.1f}, {np.nanmin(data_map['nitrate']):.1f}, {np.nanmax(data_map['nitrate']):.1f}")