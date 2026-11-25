import os
import sys
import glob
import yaml
import shutil
import warnings
import numpy as np
import xarray as xr
import taichi as ti
from src.utils import dims_ds,slice_ds
from src.simulator import OfflineSimulator

def _load_local_var(filepath, varname):
    # Check if the path is a glob pattern (contains wildcards)
    if '*' in filepath or '?' in filepath:
        # Check if the pattern actually matches any files
        if not glob.glob(filepath):
            raise FileNotFoundError(f"No files matched the pattern: {filepath}")
        
        ds = xr.open_mfdataset(filepath, engine='h5netcdf', parallel=True, chunks={'time': 10})
    else:
        # It's a single file
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Missing input file: {filepath}")
            
        ds = xr.open_dataset(filepath, engine='h5netcdf')  

    return ds[varname]

def run_simulation(cfg, config_file_path, run_name):
    # Initialize Taichi (Allows config to toggle GPU/CPU)
    if cfg.get('USE_GPU', False) == True:
        ti.init(arch=ti.metal, default_fp=ti.f32, fast_math=True)
    else:
        if not cfg['CPU_THREADS']:
            # YAML did not provide it, so we leave the argument out entirely
            ti.init(arch=ti.cpu, default_fp=ti.f32, fast_math=True)
        else:
            # YAML provided a number, so we pass the argument
            ti.init(arch=ti.cpu, default_fp=ti.f32, fast_math=True, cpu_max_num_threads=cfg['CPU_THREADS'])

    lon0, lon1 = cfg['LON_BOUNDS']
    lat0, lat1 = cfg['LAT_BOUNDS']
    dep0, dep1 = cfg['DEPTH_BOUNDS']
    date0, date1 = cfg['DATE_BOUNDS']
    
    lat_range = slice(lat0, lat1)
    lon_range = slice(lon0, lon1)
    depth_range = slice(dep0, dep1)
    time_range = slice(date0, date1)

    print(f"Loading forcing data...")

    # 1. Check which wind inputs have valid file paths
    # Extract the FORCING dictionary first
    forcing_cfg = cfg.get('FORCING', {})
    
    # Check the nested keys
    wind_defined = forcing_cfg.get('WIND', {}).get('FILE') is not None
    wind_u_defined = forcing_cfg.get('WIND_U', {}).get('FILE') is not None
    wind_v_defined = forcing_cfg.get('WIND_V', {}).get('FILE') is not None
    wind_uv_defined = wind_u_defined and wind_v_defined

    # 2. Check for missing U or V pairs
    if wind_u_defined != wind_v_defined:
        raise ValueError("Configuration Error: WIND_U and WIND_V must both be defined together.")

    # 3. Throw an error if BOTH or NEITHER are defined
    if wind_defined and wind_uv_defined:
        raise ValueError(
            "Configuration Error: Both WIND and WIND_U/WIND_V are defined. "
            "Provide either the total wind magnitude OR the U/V components, not both."
        )
    elif not wind_defined and not wind_uv_defined:
        warnings.warn(
                "No wind forcing defined in configuration. Wind fields will default to zero, "
                "resulting in zero air-sea gas flux for this simulation."
            )

    # 1. Create empty dictionary
    datasets = {}

    # 2. LOAD AND SLICE ALL VARIABLES FIRST
    for var_key, var_data in cfg['FORCING'].items():
        file_path = var_data.get('FILE', False)
        internal_name = var_data.get('NAME', False)
        
        if not file_path:
            datasets[var_key] = None
            continue
        
        print(f"Loading and subsetting {var_key} from {file_path}...")
        ds_raw = _load_local_var(file_path, internal_name)
        ds_dimmed = dims_ds(ds_raw, cfg)
        
        # SLICE IMMEDIATELY (Reduces memory footprint massively before interpolation)
        # We use .sel with method='nearest' for time to be safe, but exact for space
        try:
            ds_sliced = ds_dimmed.sel(
                time=time_range, lat=lat_range, lon=lon_range, 
                # Only slice depth if the dimension exists in this variable
                **({'depth': depth_range} if 'depth' in ds_dimmed.dims else {})
            )
            datasets[var_key] = ds_sliced
        except Exception as e:
            raise ValueError(f"Failed to slice {var_key}. Check if bounds exist in data. Error: {e}")

    # 3. CALCULATE WIND MAGNITUDE (Outside the loop, after U and V are loaded)
    if datasets.get("WIND_U") is not None and datasets.get("WIND_V") is not None:            
        datasets["WIND"] = np.sqrt(datasets["WIND_U"]**2 + datasets["WIND_V"]**2)
        print("Calculated WIND magnitude from U and V components.")

    # 4. ALIGN ATMOSPHERIC DATA TO OCEAN DATA (Outside the loop)
    # We use Temperature ('T') as the gold standard "ocean" grid reference
    ref_ocean = datasets.get('T')
    if ref_ocean is not None:
        atmos_vars = ['SW', 'WIND', 'ICE']
        for var in atmos_vars:
            ds_atm = datasets.get(var)
            if ds_atm is not None:
                if var == 'ICE':
                    print("Processing sea ice...")
                    # 1. Fix NaNs before interpolation
                    ds_atm = ds_atm.fillna(0)

                    # 2. Check metadata first (bulletproof for the 0.9% edge case)
                    units = ds_atm.attrs.get('units', '').strip().lower()
                    
                    if units in ['%', 'percent', 'percentage']:
                        print("  -> Metadata indicates percentage. Converting to fraction...")
                        ds_atm = ds_atm / 100.0
                        ds_atm.attrs['units'] = '1' # Update attribute for safety

                    # 3. Fallback: No metadata, but values are clearly > 1
                    elif ds_atm.max().item() > 1.01:
                        print("  -> Values > 1 detected. Assuming percentage and converting to fraction...")
                        ds_atm = ds_atm / 100.0
                        
                    # 4. Fallback warning for the ambiguous case
                    else:
                        print("  -> Assuming sea ice is already a fraction (0-1).")
                
                # To be deleted (leave it for sensitivity test for now)
                #if ds_atm.sizes['time'] != ref_ocean.sizes['time']:
                #    print(f"Temporal mismatch in {var}: Resampling to daily...")
                #    ds_atm = ds_atm.resample(time="1D").mean()
                
                # Force time alignment to fix 00:00 vs 12:00 offsets
                #ds_atm['time'] = ref_ocean['time']
                
                # Spatial Check
                if ds_atm.sizes['lat'] != ref_ocean.sizes['lat'] or ds_atm.sizes['lon'] != ref_ocean.sizes['lon']:
                    print(f"Spatial mismatch in {var}: Interpolating to ocean grid...")
                    ds_atm = ds_atm.interp(lat=ref_ocean.lat, lon=ref_ocean.lon, method='linear'
                                          ).ffill(dim="lat").bfill(dim="lat").ffill(dim="lon").bfill(dim="lon")
                    
                
                datasets[var] = ds_atm
    
    # BGC Inputs (remove time dimension via suqeeze())
    ds_clim = xr.open_dataset(cfg['CLIM_FILE']).squeeze() if cfg['CLIM_FILE'] else None
    ds_clim = dims_ds(ds_clim, cfg) if cfg['CLIM_FILE'] else None
    ds_restart = xr.open_dataset(cfg['RESTART_FILE']).squeeze() if cfg['RESTART_FILE'] else None
    ds_restart = dims_ds(ds_restart, cfg) if cfg['RESTART_FILE'] else None
    
    save_flags = {key: val.get('SAVE', False) for key, val in cfg['FORCING'].items()}

    # Initialize Simulator
    sim = OfflineSimulator(
        bgc_model_choice=cfg['BGC_MODEL'], exp_name=run_name,
        dt_phys=cfg['DT_SEC'], num_uv_filter=cfg.get('NUM_UV_FILTER', 0), sponge_width=cfg.get('SPONGE_WIDTH', 0),
        tau_lateral=cfg['TAU_LATERAL'], tau_bottom=cfg['TAU_BOTTOM'], 
        tau_coast=cfg['TAU_COAST'], is_global=cfg.get('IS_GLOBAL', False),
        forcing_save_flags=save_flags
    )

    sim.prepare_forcing(ds_forcing=datasets, ds_clim=ds_clim, ds_restart=ds_restart)

    shutil.copy(config_file_path, os.path.join(sim.out_dir, os.path.basename(config_file_path)))
    sim.run()

# ==========================================
# COMMAND LINE INTERFACE
# ==========================================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run.py <path_to_config.yaml>")
        sys.exit(1)

    config_file_path = sys.argv[1]

    if not os.path.exists(config_file_path):
        print(f"Error: Could not find configuration file at '{config_file_path}'")
        sys.exit(1)

    with open(config_file_path, 'r') as file:
        cfg = yaml.safe_load(file)

    # --- NEW LOGIC HERE ---
    # 1. os.path.basename removes "configs/" -> "hokkaido_era5.yml"
    # 2. os.path.splitext removes ".yml" -> "hokkaido_era5"
    run_name = os.path.splitext(os.path.basename(config_file_path))[0]
    
    # Dynamically overwrite the EXP_NAME in the dictionary
    print(f"Starting experiment: {run_name}")

    # Run the simulation
    run_simulation(cfg, config_file_path, run_name)
