import numpy as np
import taichi as ti
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
                 is_global=False,
                 forcing_save_flags=None):
        
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
        self.forcing_save_flags = forcing_save_flags or {}      
        
        self.bgc_model = None
        self.ds_clim = None

    def prepare_forcing(self, lat_range, lon_range, depth_range, time_range, 
                        ds_forcing, ds_clim=None, ds_restart=None, glodap_dir=None):
        print("Preparing and subsetting forcing datasets...")
        
        self.ds_t = ds_forcing["T"].sel(time=time_range, depth=depth_range, lat=lat_range, lon=lon_range)
        self.ds_s = ds_forcing["S"].sel(time=time_range, depth=depth_range, lat=lat_range, lon=lon_range)
        self.ds_u = ds_forcing["U"].sel(time=time_range, depth=depth_range, lat=lat_range, lon=lon_range)
        self.ds_v = ds_forcing["V"].sel(time=time_range, depth=depth_range, lat=lat_range, lon=lon_range)
        self.ds_sw = ds_forcing["SW"].sel(time=time_range, lat=lat_range, lon=lon_range)

        if ds_forcing["K"] is None:
            self.ds_k = None
        else:
            self.ds_k = ds_forcing["K"].sel(time=time_range, depth=depth_range, lat=lat_range, lon=lon_range)

        if ds_forcing["WIND"] is None:
            self.ds_wind = xr.zeros_like(self.ds_sw)
        else:
            self.ds_wind = ds_forcing["WIND"].sel(time=time_range, lat=lat_range, lon=lon_range)

        if ds_forcing["ICE"] is None:
            self.ds_ice = xr.ones_like(self.ds_sw)
        else:
            self.ds_ice = ds_forcing["ICE"].sel(time=time_range, lat=lat_range, lon=lon_range)

        if ds_clim is not None:
            self.ds_clim = ds_clim.sel(depth=depth_range, lat=lat_range, lon=lon_range)
        elif glodap_dir is not None:
            print("\nGenerating BGC climatology on-the-fly from GLODAP...")
            ref_template = self.ds_t.isel(time=0).squeeze()
            self.ds_clim = generate_restoring_climatology(ref_template, glodap_dir)
        else:
            self.ds_clim = None

        if ds_restart is not None:
            self.ds_restart = ds_restart.squeeze().sel(depth=depth_range, lat=lat_range, lon=lon_range)
        else: self.ds_restart = None

        # Pre-load data entirely into Memory
        print("Loading subsetted forcing data directly into RAM to bypass disk I/O...")
        
        # 1. Calculate total memory needed
        total_bytes = (
            self.ds_t.nbytes + self.ds_s.nbytes + self.ds_u.nbytes + 
            self.ds_v.nbytes + self.ds_sw.nbytes + self.ds_wind.nbytes + self.ds_ice.nbytes
        )
        total_gb = total_bytes / (1024 ** 3)
        print(f"  -> MEMORY CHECK: Total RAM required is ~{total_gb:.2f} GB")
        
        # Prevent Mac from crashing (Adjust the 12.0 GB limit based on your machine!)
        if total_gb > 12.0:
            print("  -> [ERROR] Dataset is too massive! Reduce DATE_BOUNDS or spatial bounds in config.yml.")
            exit()

        # 2. Load with progress tracking
        print("  -> Loading T...")
        self.ds_t.load()
        print("  -> Loading S...")
        self.ds_s.load()
        print("  -> Loading U...")
        self.ds_u.load()
        print("  -> Loading V...")
        self.ds_v.load()
        print("  -> Loading SW...")
        self.ds_sw.load()
        print("  -> Loading WIND...")
        self.ds_wind.load()
        print("  -> Loading ICE...")
        self.ds_ice.load()
        
        if getattr(self, 'ds_k', None) is not None: 
            print("  -> Loading K...")
            self.ds_k.load()
        if getattr(self, 'ds_clim', None) is not None:
            print("  -> Loading Climatology...")
            self.ds_clim.load()
        if getattr(self, 'ds_restart', None) is not None:
            print("  -> Loading Restart...")
            self.ds_restart.load()
            
        print("Data successfully loaded into RAM!")

        self.setup_grid()
        
        print(f"Initializing BGC Model: {self.bgc_model_choice}...")
        self.bgc_model = bgc_models.get_model(self.bgc_model_choice, self.nz, self.ny, self.nx, self.water_mask, params=self.bgc_params)
        self.bgc_model.initialize(ds_restart=ds_restart, ds_clim=self.ds_clim)
        if ds_restart is not None:
            ds_restart.close()

        self.setup_io()
        self.setup_restoring()

    def setup_grid(self):
        print("Initializing Grid...")
        if self.ds_t['lat'].ndim == 2:
            self.dx, self.dy = physics.calculate_metrics_curvilinear(self.ds_t['lon'].values, self.ds_t['lat'].values)
            self.lon_2d = self.ds_t['lon'].values
            self.lat_2d = self.ds_t['lat'].values
        else:
            self.dx, self.dy = physics.calculate_grid_metrics(self.ds_t)
            self.lon_2d, self.lat_2d = np.meshgrid(self.ds_t['lon'].values, self.ds_t['lat'].values)
            
        self.nz, self.ny, self.nx = self.ds_t.depth.size, self.ds_t.lat.size, self.ds_t.lon.size
        dz_1d = physics.calculate_dz_from_centers(self.ds_t['depth'].values)
        self.dz_3d = np.tile(dz_1d[:, None, None], (1, self.ny, self.nx))
        self.p_3d = self.ds_t['depth'].values[:, None, None]
        
        ref_val = self.ds_t.isel(time=0).values if 'time' in self.ds_t.dims else self.ds_t.values
        self.water_mask = ~np.isnan(ref_val) 
        self.dz_static = np.where(self.water_mask, self.dz_3d, 0.0)
        
        print("Generating Restoring Map...")
        self.nudge_map = physics.create_restoring_weights(
            self.water_mask, self.sponge_width, self.dt_phys, self.tau_lateral, self.tau_bottom, self.tau_coast, self.is_global
        ).astype(np.float32)
        
        self.allocate_taichi_memory()

    def allocate_taichi_memory(self):
        print("Allocating GPU Memory (Taichi Fields)...")
        # 3D Arrays
        shape_3d = (self.nz, self.ny, self.nx)
        self.ti_u = ti.field(dtype=ti.f32, shape=shape_3d)
        self.ti_v = ti.field(dtype=ti.f32, shape=shape_3d)
        self.ti_w = ti.field(dtype=ti.f32, shape=(self.nz + 1, self.ny, self.nx))        
        self.ti_rho = ti.field(dtype=ti.f32, shape=shape_3d)
        self.ti_kz = ti.field(dtype=ti.f32, shape=shape_3d)
        self.ti_dz = ti.field(dtype=ti.f32, shape=shape_3d)
        self.ti_tracer = ti.field(dtype=ti.f32, shape=shape_3d)
        self.ti_tracer_new = ti.field(dtype=ti.f32, shape=shape_3d)
        
        # 2D Arrays
        shape_2d = (self.ny, self.nx)
        self.ti_dx = ti.field(dtype=ti.f32, shape=shape_2d)
        self.ti_dy = ti.field(dtype=ti.f32, shape=shape_2d)
        self.ti_mld = ti.field(dtype=ti.f32, shape=shape_2d)
        
        # O2 specific 2D fields
        self.ti_o2_surf = ti.field(dtype=ti.f32, shape=shape_2d)
        self.ti_o2_sat = ti.field(dtype=ti.f32, shape=shape_2d)
        self.ti_temp_surf = ti.field(dtype=ti.f32, shape=shape_2d)
        self.ti_wind_surf = ti.field(dtype=ti.f32, shape=shape_2d)
        self.ti_ice_surf = ti.field(dtype=ti.f32, shape=shape_2d)
        self.ti_dz_surf = ti.field(dtype=ti.f32, shape=shape_2d)

        # Pre-load the static geometry fields onto the GPU
        self.ti_dz.from_numpy(self.dz_static.astype(np.float32))
        self.ti_dx.from_numpy(self.dx.astype(np.float32))
        self.ti_dy.from_numpy(self.dy.astype(np.float32))
        self.ti_dz_surf.from_numpy(self.dz_static[0, :, :].astype(np.float32))

    def setup_io(self):
        self.ds_template = self.ds_t.isel(time=0).drop_vars('time')
        self.out_dir = f"output/{self.exp_name}/{self.bgc_model_choice}"
        os.makedirs(self.out_dir, exist_ok=True)

        static_grid_file = f"{self.out_dir}/static_grid_{self.exp_name}_{self.bgc_model_choice}.nc"
        if not os.path.exists(static_grid_file):
            print("Generating Static Grid Metrics File...")
            ds_grid = physics.generate_static_grid(self.ds_template)
            ds_grid.to_netcdf(static_grid_file)

    def setup_restoring(self):
        self.restoring_data = {}
        if self.ds_clim is None:
            return
            
        print("Loading Restoring Targets (Sponge)...")
        for model_var in self.bgc_model.names:
            clim_var = GLODAP_MAP.get(model_var, model_var)
            if clim_var in self.ds_clim:
                self.restoring_data[model_var] = np.nan_to_num(self.ds_clim[clim_var].values).astype(np.float32)

    def run(self): 
        nt = self.ds_t['time'].size
        print(f"Starting Simulation ({nt} days)...")
        
        for day in range(nt):
            print(f"  Day {day+1} / {nt}")
            
            # 1. Load Forcing 
            t = np.nan_to_num(self.ds_t.isel(time=day).values).copy()
            s = np.nan_to_num(self.ds_s.isel(time=day).values).copy()            
            u = np.nan_to_num(self.ds_u.isel(time=day).values).copy()
            v = np.nan_to_num(self.ds_v.isel(time=day).values).copy()
            s = np.maximum(s, 0.0)
            
            sw = np.nan_to_num(self.ds_sw.isel(time=day).values).copy()
            wind_surf = np.nan_to_num(self.ds_wind.isel(time=day).values).copy() 
            ice_surf = np.nan_to_num(self.ds_ice.isel(time=day).values).copy()            
            
            # 2. Physics & State Setup
            SA = gsw.SA_from_SP(s, self.p_3d, self.lon_2d, self.lat_2d)
            CT = gsw.CT_from_pt(SA, t)
            rho_3d = np.where(self.water_mask, gsw.sigma0(SA, CT), 0.0)
            u[~self.water_mask] = 0.0
            v[~self.water_mask] = 0.0

            # --- Push daily arrays to the GPU ---
            self.ti_u.from_numpy(u.astype(np.float32))
            self.ti_v.from_numpy(v.astype(np.float32))
            self.ti_rho.from_numpy(rho_3d.astype(np.float32))
            
            # --- Execute Taichi Physics Kernels ---
            physics.calculate_w_rigid_lid_ti(self.ti_u, self.ti_v, self.ti_dz, self.ti_dx, self.ti_dy, self.ti_w)
            physics.calculate_mld_ti(self.ti_rho, self.ti_dz, self.ti_mld)
            
            if self.ds_k is None:
                physics.calculate_full_kz_ti(self.ti_mld, self.ti_rho, self.ti_dz, self.ti_kz, 1e-2, 1e-6, 1e-5, 1e-4)
            else:
                k_val = np.nan_to_num(self.ds_k.isel(time=day).values).copy()
                self.ti_kz.from_numpy(k_val.astype(np.float32))

            # Pull mld and kz back to CPU for I/O saving later
            mld_2d_np = self.ti_mld.to_numpy()
            kz_3d_np = self.ti_kz.to_numpy()
            
            if "oxygen" in self.bgc_model.tracers:
                o2_sat_umol_kg = gsw.O2sol_SP_pt(s[0, :, :], t[0, :, :])
                o2_sat_mmol_m3 = o2_sat_umol_kg * (rho_3d[0, :, :] / 1000.0)
            
            is_glob_int = 1 if self.is_global else 0

            # --- SUB-STEPPED LOOP ---
            for step in range(self.steps_per_day):
                
                # A. PHYSICS (TAICHI GPU)
                for name, tr_data in self.bgc_model.tracers.items():
                    # Stream CPU array to GPU
                    self.ti_tracer.from_numpy(tr_data.astype(np.float32))
                    
                    # Compute advection & diffusion directly on the GPU
                    physics.advection_neumann_ti(self.ti_tracer, self.ti_tracer_new, self.ti_u, self.ti_v, self.ti_w, self.ti_dz, self.ti_dx, self.ti_dy, float(self.dt_phys), is_glob_int)
                    physics.diffusion_robust_ti(self.ti_tracer_new, self.ti_kz, self.ti_dz, float(self.dt_phys))
                        
                    # Stream updated array back to CPU and clamp
                    self.bgc_model.tracers[name] = np.maximum(self.ti_tracer_new.to_numpy(), 0.0)

                # B. RESTORING (TAICHI)
                for var_name, clim_data in self.restoring_data.items():
                    physics.apply_restoring_ti(
                        self.bgc_model.tracers[var_name], 
                        clim_data, 
                        self.nudge_map
                    )

                # C. BIOLOGY & SINKING (NUMBA CPU)
                par_3d = self.bgc_model.biology_step(t, sw, self.dz_static, self.dt_phys)
                self.bgc_model.sinking_step(self.dz_static, self.dt_phys)

                for name, tr_data in self.bgc_model.tracers.items():
                    self.bgc_model.tracers[name] = np.maximum(tr_data, 0.0)

                # D. AIR-SEA FLUX (TAICHI GPU)
                if "oxygen" in self.bgc_model.tracers:
                    self.ti_o2_surf.from_numpy(self.bgc_model.tracers["oxygen"][0, :, :].astype(np.float32))
                    self.ti_o2_sat.from_numpy(o2_sat_mmol_m3.astype(np.float32))
                    self.ti_temp_surf.from_numpy(t[0, :, :].astype(np.float32))
                    self.ti_wind_surf.from_numpy(wind_surf.astype(np.float32))
                    self.ti_ice_surf.from_numpy(ice_surf.astype(np.float32))
                    
                    physics.calc_o2_flux_ti(self.ti_o2_surf, self.ti_o2_sat, self.ti_temp_surf, self.ti_wind_surf, self.ti_ice_surf, self.ti_dz_surf, float(self.dt_phys))
                    
                    self.bgc_model.tracers["oxygen"][0, :, :] = np.maximum(self.ti_o2_surf.to_numpy(), 0.0)
                
            # Save Output
            # Note: We pass the pulled NumPy mld and kz arrays
            self.save_day(day, self.ds_t.isel(time=day).time.values, par_3d, rho_3d, kz_3d_np, t, s, u, v, mld_2d_np, sw, wind_surf, ice_surf)

    def save_day(self, day, current_time, par_3d, rho_3d, kz_3d_np, t, s, u, v, mld_2d_np, sw, wind_surf, ice_surf):
        
        data_map = {name: arr for name, arr in self.bgc_model.tracers.items()}
        data_map['PAR'] = par_3d
        ds_out = xr.Dataset(coords=self.ds_template.coords)
        
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
        
        ds_out['sigma0'] = (self.ds_template.dims, np.where(self.water_mask, rho_3d, np.nan))
        ds_out['sigma0'].attrs = {'units': 'kg/m3', 'long_name': 'Potential Density Anomaly (Sigma-0)'}

        ds_out['MLD'] = (dims_2d, np.where(mask_2d, mld_2d_np, np.nan))
        ds_out['MLD'].attrs = {'units': 'm', 'long_name': 'Mixed Layer Depth'}

        forcing_map = {
            'T': {'data': t, 'units': 'celcius', 'long_name': 'Temperature'},
            'S': {'data': s, 'units': 'psu', 'long_name': 'Salinity'},
            'U': {'data': u, 'units': 'm/s', 'long_name': 'Zonal velocity'},
            'V': {'data': v, 'units': 'm/s', 'long_name': 'Meridional velocity'},
            'K': {'data': kz_3d_np, 'units': 'm2/s', 'long_name': 'Vertical eddy diffusivity'},
            'SW': {'data': sw, 'units': 'W/m2', 'long_name': 'Shortwave radiation'},
            'WIND': {'data': wind_surf, 'units': 'm/s', 'long_name': 'Wind speed'},
            'ICE': {'data': ice_surf, 'units': '-', 'long_name': 'Sea ice concentration'}
        }

        for yaml_key, meta in forcing_map.items():
            # Check the dictionary stored in __init__
            if self.forcing_save_flags.get(yaml_key, False):
                ds_out[yaml_key] = (self.ds_template.dims, np.where(self.water_mask, meta['data'], np.nan))
                ds_out[yaml_key].attrs = {'units': meta['units'], 'long_name': meta['long_name']}

        ds_out = ds_out.expand_dims(time=[current_time])
        t_str = pd.to_datetime(current_time).strftime('%Y%m%d')
        fname = f"{self.out_dir}/output_{self.exp_name}_{self.bgc_model_choice}_{t_str}.nc"
        
        if os.path.exists(fname):
            try: os.remove(fname)  
            except PermissionError:
                print(f"  [Error] Cannot overwrite {fname}. Skipping save.")
                return 
        # Save. Note we do not compress because it is a bottleneck.
        # Apply compression as post-processing if necessary.
        ds_out.to_netcdf(fname)
        
        print(f"    -> Saved {fname}")
        if 'nitrate' in data_map:
            nit_np = np.asarray(data_map['nitrate'])
            print(f"    -> Nitrate Mean, Min, Max: {np.nanmean(nit_np):.1f}, {np.nanmin(nit_np):.1f}, {np.nanmax(nit_np):.1f}")