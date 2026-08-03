import jax.numpy as jnp
import numpy as np
import xarray as xr
import pandas as pd
import os
import gsw
from . import physics
from . import bgc_models
from .bgc_models.utils import GLODAP_MAP
from .climatology import generate_restoring_climatology

class OfflineSimulator:
    def __init__(self, 
                 bgc_model_choice, 
                 exp_name, 
                 dt_phys, 
                 bgc_params=None,
                 sponge_width=5, 
                 tau_lateral=432000.0, 
                 tau_bottom=5184000.0,
                 tau_coast=432000.0, 
                 is_global=False):
        
        # 1. Store settings
        self.bgc_model_choice = bgc_model_choice
        self.exp_name = exp_name
        self.dt_phys = dt_phys
        self.steps_per_day = int(86400 / self.dt_phys)
        self.bgc_params = bgc_params
        self.sponge_width = sponge_width
        self.tau_lateral = tau_lateral
        self.tau_bottom = tau_bottom
        self.tau_coast = tau_coast
        self.is_global = is_global
        
        # These will be set during prepare_forcing()
        self.bgc_model = None
        self.ds_clim = None

    def prepare_forcing(self, 
                        lat_range, 
                        lon_range, 
                        depth_range, 
                        time_range, 
                        ds_t, 
                        ds_s, 
                        ds_u, 
                        ds_v, 
                        ds_sw, 
                        ds_wind, 
                        ds_ice=None, 
                        ds_k=None, 
                        ds_clim=None, 
                        ds_restart=None,
                        glodap_dir=None
                       ):
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
            print("  -> No Kz file provided. Using dummy array and parameterize Kz.")
            self.ds_k = None
        else:
            self.ds_k = ds_k.sel(time=time_range, depth=depth_range, lat=lat_range, lon=lon_range)

        if ds_wind is None:
            print("  -> No wind file provided. Setting all to ZERO (no gas exchange).")
            self.ds_wind = xr.zeros_like(self.ds_sw)
        else:
            self.ds_wind = ds_wind.sel(time=time_range, lat=lat_range, lon=lon_range)

        if ds_ice is None:
            print("  -> No Ice file provided. Setting all to ONE (no ice cover).")
            self.ds_ice = xr.ones_like(self.ds_sw)
        else:
            self.ds_ice = ds_ice.sel(time=time_range, lat=lat_range, lon=lon_range)

    # --- 3. BGC CLIMATOLOGY & RESTART ---
        if ds_clim is not None:
            # User provided a pre-processed climatology file
            self.ds_clim = ds_clim.sel(depth=depth_range, lat=lat_range, lon=lon_range)
        elif glodap_dir is not None:
            # User provided a directory; generate it on the fly!
            print("\nGenerating BGC climatology on-the-fly from GLODAP...")
            
            # Use the first timestep of our already-subsetted temperature grid as the exact template
            ref_template = self.ds_t.isel(time=0).squeeze()
            
            # Call your new function
            self.ds_clim = generate_restoring_climatology(ref_template, glodap_dir)
        else:
            self.ds_clim = None

        if ds_restart is not None:
            ds_restart = ds_restart.squeeze().sel(depth=depth_range, lat=lat_range, lon=lon_range)

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
            self.water_mask, self.sponge_width, self.dt_phys, self.tau_lateral, self.tau_bottom, self.tau_coast, self.is_global
        )

    def setup_io(self):
        self.ds_template = self.ds_t.isel(time=0).drop_vars('time')
        self.out_dir = f"output/{self.exp_name}/{self.bgc_model_choice}"
        os.makedirs(self.out_dir, exist_ok=True)

        # --- NEW: Generate and save static grid metrics ---
        static_grid_file = f"{self.out_dir}/static_grid_{self.exp_name}_{self.bgc_model_choice}.nc"
        
        if not os.path.exists(static_grid_file):
            print("Generating Static Grid Metrics File...")
            ds_grid = physics.generate_static_grid(self.ds_template)
            
            # Save it once. 
            ds_grid.to_netcdf(static_grid_file)
            print(f"  -> Saved {static_grid_file}")

        
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
            t = np.nan_to_num(self.ds_t.isel(time=day).values)
            s = np.nan_to_num(self.ds_s.isel(time=day).values)            
            u = np.nan_to_num(self.ds_u.isel(time=day).values)
            v = np.nan_to_num(self.ds_v.isel(time=day).values)

            # set negative salinity to zero (e.g. JCOPE2M)
            s = s = np.maximum(s, 0.0)
            sw = np.nan_to_num(self.ds_sw.isel(time=day).values)
            wind_surf = np.nan_to_num(self.ds_wind.isel(time=day).values) 
            ice_surf = np.nan_to_num(self.ds_ice.isel(time=day).values)            
            
            # 2. Calculate potential density anomaly and MLD           
            SA = gsw.SA_from_SP(s, self.p_3d, self.lon_2d, self.lat_2d)
            CT = gsw.CT_from_pt(SA, t)
            rho_3d = gsw.sigma0(SA, CT)
            
            # Re-mask the calculated density back to 0.0 over land
            rho_3d = np.where(self.water_mask, rho_3d, 0.0)
            
            self.mld_2d = physics.calculate_mld(rho_3d, self.dz_static)
            
            if self.ds_k is None:
                k = physics.calculate_full_kz_jax(self.mld_2d, rho_3d, self.dz_3d)
            else:
                k = np.nan_to_num(self.ds_k.isel(time=day).values)
            
            # Calc W
            u[~self.water_mask] = 0.0
            v[~self.water_mask] = 0.0
            w = physics.calculate_w_rigid_lid(u, v, self.dz_static, self.dx, self.dy)  
            
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
                    # The @jax.jit decorator automatically compiles the math 
                    # on the very first time step, making the remaining steps incredibly fast.
                    tr_adv = physics.advection_neumann_jax(tr_data, u, v, w, self.dz_static, self.dt_phys, self.dx, self.dy, is_global=self.is_global)
                    tr_mix = physics.diffusion_robust_jax(tr_adv, k, self.dz_static, self.dt_phys)
                        
                    # CLAMP #1: Immediately after physics to fix advection overshoots (JAX compliant)
                    self.bgc_model.tracers[name] = jnp.maximum(tr_mix, 0.0)

                # B. RESTORING
                for var_name, clim_data in self.restoring_data.items():
                    diff = clim_data - self.bgc_model.tracers[var_name]
                    # FIX: Cannot use += on JAX arrays. Must reassign.
                    self.bgc_model.tracers[var_name] = self.bgc_model.tracers[var_name] + (diff * self.nudge_map)

                # C. BIOLOGY & SINKING
                par_3d = self.bgc_model.biology_step(t, sw, self.dz_static, self.dt_phys)
                self.bgc_model.sinking_step(self.dz_static, self.dt_phys)

                # CLAMP #2: Immediately after biology/sinking (JAX compliant)
                for name, tr_data in self.bgc_model.tracers.items():
                    self.bgc_model.tracers[name] = jnp.maximum(tr_data, 0.0)

                # D. AIR-SEA FLUX (OXYGEN)
                if "oxygen" in self.bgc_model.tracers:
                    o2_surf = self.bgc_model.tracers["oxygen"][0, :, :]
                    temp_surf = t[0, :, :]
                    dz_surf = self.dz_static[0, :, :]
                    
                    # FIX: Called the _jax function
                    updated_o2_surf = physics.calc_o2_flux(
                        o2_surf, o2_sat_mmol_m3, temp_surf, wind_surf, 
                        ice_surf, dz_surf, self.dt_phys
                    )
                    # FIX: JAX slice reassignment
                    self.bgc_model.tracers["oxygen"] = self.bgc_model.tracers["oxygen"].at[0, :, :].set(
                        jnp.maximum(updated_o2_surf, 0.0)
                    )
                
            # Save Output              
            self.save_day(day, self.ds_t.isel(time=day).time.values, par_3d, rho_3d, k, t, s, u, v)

    def save_day(self, day, current_time, par_3d, rho_3d, k, t, s, u, v):
        
        data_map = {name: arr for name, arr in self.bgc_model.tracers.items()}
        data_map['PAR'] = par_3d
        ds_out = xr.Dataset(coords=self.ds_template.coords)
        
        # This loop handles tracers and PAR
        for var, arr in data_map.items():
            arr_np = np.asarray(arr)
            ds_out[var] = (self.ds_template.dims, np.where(self.water_mask, arr_np, np.nan))
            if var == 'PAR':
                ds_out[var].attrs = {'units': 'W/m2', 'long_name': 'Photosynthetically Active Radiation'}
            elif 'chl' in var:
                ds_out[var].attrs = {'units': 'mg/m3'}                  
            else:
                ds_out[var].attrs = {'units': 'mmol/m3'}  
                
        dims_2d = [dim for dim in self.ds_template.dims if dim != 'depth']
        mask_2d = self.water_mask[0, :, :] 
        
        rho_np = np.asarray(rho_3d)
        ds_out['sigma0'] = (self.ds_template.dims, np.where(self.water_mask, rho_np, np.nan))
        ds_out['sigma0'].attrs = {'units': 'kg/m3', 'long_name': 'Potential Density Anomaly (Sigma-0)'}

        # --- Save Diffusivity ---
        kz_np = np.asarray(k)
        ds_out['kz'] = (self.ds_template.dims, np.where(self.water_mask, kz_np, np.nan))
        ds_out['kz'].attrs = {'units': 'm2/s', 'long_name': 'Vertical Eddy Diffusivity'}

        # --- Save Diffusivity ---
        t_np = np.asarray(t)
        ds_out['t'] = (self.ds_template.dims, np.where(self.water_mask, t_np, np.nan))
        ds_out['t'].attrs = {'units': 'celcius', 'long_name': 'Temperature'}
        s_np = np.asarray(s)
        ds_out['s'] = (self.ds_template.dims, np.where(self.water_mask, s_np, np.nan))
        ds_out['s'].attrs = {'units': 'psu', 'long_name': 'Salinity'}
        u_np = np.asarray(u)
        ds_out['u'] = (self.ds_template.dims, np.where(self.water_mask, u_np, np.nan))
        ds_out['u'].attrs = {'units': 'm/s', 'long_name': 'Zonal velocity'}
        v_np = np.asarray(v)
        ds_out['v'] = (self.ds_template.dims, np.where(self.water_mask, v_np, np.nan))
        ds_out['v'].attrs = {'units': 'm/s', 'long_name': 'Meridional velocity'}

        # Pull mld back to numpy
        mld_np = np.asarray(self.mld_2d)
        ds_out['MLD'] = (dims_2d, np.where(mask_2d, mld_np, np.nan))
        ds_out['MLD'].attrs = {'units': 'm', 'long_name': 'Mixed Layer Depth'}
            
        ds_out = ds_out.expand_dims(time=[current_time])
        t_str = pd.to_datetime(current_time).strftime('%Y%m%d')
        fname = f"{self.out_dir}/output_{self.exp_name}_{self.bgc_model_choice}_{t_str}.nc"
        
        if os.path.exists(fname):
            try: os.remove(fname)  
            except PermissionError:
                print(f"  [Error] Cannot overwrite {fname}. Skipping save.")
                return 
                
        comp = dict(zlib=True, complevel=1)
        enc = {v: comp for v in ds_out.data_vars}
        ds_out.to_netcdf(fname, encoding=enc)
        
        print(f"    -> Saved {fname}")
        if 'nitrate' in data_map:
            # Use numpy on the pulled array
            nit_np = np.asarray(data_map['nitrate'])
            print(f"    -> Nitrate Mean, Min, Max: {np.nanmean(nit_np):.1f}, {np.nanmin(nit_np):.1f}, {np.nanmax(nit_np):.1f}")