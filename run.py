import os
import sys
import glob
import yaml
import shutil
import numpy as np
import xarray as xr
import taichi as ti

from source.simulator import OfflineSimulator

def _load_local_var(filepath, varname):
    # Check if the path is a glob pattern (contains wildcards)
    if '*' in filepath or '?' in filepath:
        # Check if the pattern actually matches any files
        if not glob.glob(filepath):
            raise FileNotFoundError(f"No files matched the pattern: {filepath}")
        
        ds = xr.open_mfdataset(filepath)
    else:
        # It's a single file
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Missing input file: {filepath}")
            
        ds = xr.open_dataset(filepath)
        
    # Auto-rename common weird dimension names from raw files
    rename_map = {"longitude": "lon", 
                  "latitude": "lat", 
                  "valid_time": "time"
                 }
    rename_dict = {k: v for k, v in rename_map.items() if k in ds.dims}
    return ds.rename(rename_dict)[varname]

def run_simulation(cfg, config_file_path):
    # Initialize Taichi (Allows config to toggle GPU/CPU)
    if cfg.get('USE_GPU', False) == True:
        ti.init(arch=ti.metal, default_fp=ti.f32)
    else:
        if not cfg['MAX_THREADS']:
            # YAML did not provide it, so we leave the argument out entirely
            ti.init(arch=ti.cpu, default_fp=ti.f32)
        else:
            # YAML provided a number, so we pass the argument
            ti.init(arch=ti.cpu, default_fp=ti.f32, cpu_max_num_threads=cfg['CPU_THREADS'])

    lon0, lon1 = cfg['LON_BOUNDS']
    lat0, lat1 = cfg['LAT_BOUNDS']
    dep0, dep1 = cfg['DEPTH_BOUNDS']
    date0, date1 = cfg['DATE_BOUNDS']
    
    lat_range = slice(lat0, lat1)
    lon_range = slice(lon0, lon1)
    depth_range = slice(dep0, dep1)
    time_range = slice(date0, date1)

    print(f"Loading forcing data...")

    # 1. Create an empty dictionary to store your loaded datasets
    # (Since you are looping, you can no longer do ds_t = ..., ds_s = ...)
    datasets = {}

    # 2. Loop through the FORCING dictionary
    for var_key, var_data in cfg['FORCING'].items():
        
        # Use .get() so it defaults to False if the user forgets the line entirely
        file_path = var_data.get('FILE', False)
        internal_name = var_data.get('NAME', False)
        
        # Skip assigning data if file_path is set to false.
        if not file_path:
            datasets[var_key] = None
            continue
            
        print(f"Loading {var_key} from {file_path}...")
        
        datasets[var_key] = _load_local_var(f"input/{run_name}/{file_path}", internal_name)
  
        # Optional daily resampling for atmospheric data
#        if cfg.get('RESAMPLE_ATMOS_DAILY', False):
#            ds_sw = ds_sw.resample(time="1D").mean()
#            ds_wind = ds_wind.resample(time="1D").mean()
        
        # If U- and V-component of wind data are provided, calculate the magnitude
        if datasets.get("WIND_U") is not None and datasets.get("WIND_V") is not None:            
            datasets["WIND"] = np.sqrt(datasets["WIND_U"]**2 + datasets["WIND_V"]**2)

    # BGC Inputs
    ds_clim = xr.open_mfdataset(cfg['CLIM_FILE']) if cfg['CLIM_FILE'] else None
    ds_restart = xr.open_dataset(cfg['RESTART_FILE']).squeeze() if cfg['RESTART_FILE'] else None

    save_flags = {key: val.get('SAVE', False) for key, val in cfg['FORCING'].items()}

    # Initialize Simulator
    sim = OfflineSimulator(
        bgc_model_choice=cfg['BGC_MODEL'], exp_name=cfg['EXP_NAME'],
        dt_phys=cfg['DT_SEC'], sponge_width=cfg['SPONGE_WIDTH'],
        tau_lateral=cfg['TAU_LATERAL'], tau_bottom=cfg['TAU_BOTTOM'], 
        tau_coast=cfg['TAU_COAST'], is_global=cfg.get('IS_GLOBAL', False),
        forcing_save_flags=save_flags
    )

    sim.prepare_forcing(
        lat_range=lat_range, lon_range=lon_range, depth_range=depth_range, time_range=time_range,
        ds_forcing=datasets, ds_clim=ds_clim, glodap_dir=cfg.get('GLODAP_DIR'), ds_restart=ds_restart)

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
    cfg['EXP_NAME'] = run_name
    print(f"Starting experiment: {run_name}")

    # Run the simulation
    run_simulation(cfg, config_file_path)