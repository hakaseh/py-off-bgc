import os
# --- 0. HARDWARE LIMITS (MUST BE SET FIRST) ---
# Define exactly how many CPU cores you want to use
NUM_CORES = "8"
# 1. Restrict JAX's internal XLA compiler threadpool
os.environ["XLA_FLAGS"] = f"--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads={NUM_CORES}"
# 2. Restrict NumPy/Xarray background threadpools (highly recommended)
os.environ["OMP_NUM_THREADS"] = NUM_CORES
os.environ["OPENBLAS_NUM_THREADS"] = NUM_CORES
os.environ["MKL_NUM_THREADS"] = NUM_CORES

import sys
import shutil
import numpy as np
import xarray as xr
from source.simulator import OfflineSimulator

for year in range(2017,2024):
    # --- 1. CONFIGURATION ---
    exp_name = "BRAN2020"
    bgc_model_choice = "NEMURO"
    dt_in_sec = 1200
    sponge_choice = 1
    tau_lateral_choice = 86400.0 * 1
    tau_bottom_choice = 86400.0 * 30
    tau_coast_choice = 86400.0 * 1
    is_global_choice = False
    restart_file = f"output/{exp_name}/{bgc_model_choice}/output_{exp_name}_{bgc_model_choice}_{year-1}1231.nc"
    clim_file = None #f"climatology/{exp_name}/GLODAPv2.2016b.ALL_{exp_name}.nc"
    glodap_dir = "climatology/GLODAPv2.2016b.MappedClimatologies/"
    
    # Domain Slicing
    lat_range   = slice(None, None)
    lon_range   = slice(None, None)
    depth_range = slice(None, None)
    time_range  = slice(None, None)
    
    # --- 2. LOAD PHYSICAL DATA ---
    ds_t = xr.open_dataset(f'input/{exp_name}/nwp/input_temp_{year}_BRAN2020_2016-2023_nwp.nc')["temp"]
    ds_s = xr.open_dataset(f'input/{exp_name}/nwp/input_salt_{year}_BRAN2020_2016-2023_nwp.nc')["salt"]
    ds_u = xr.open_dataset(f'input/{exp_name}/nwp/input_u_{year}_BRAN2020_2016-2023_nwp.nc')["u"]
    ds_v = xr.open_dataset(f'input/{exp_name}/nwp/input_v_{year}_BRAN2020_2016-2023_nwp.nc')["v"]
    ds_sw = xr.open_dataset(f'input/{exp_name}/nwp/input_rsds_{year}_BRAN2020_2016-2023_nwp.nc')["rsds"]
    
    # Optional input
    ds_k = None
    ds_ice = None
    ds_wind = None
    
    # BGC Inputs
    ds_clim = xr.open_mfdataset(clim_file) if clim_file else None
    ds_restart = xr.open_dataset(restart_file) if restart_file else None
    
    # --- 3. EXECUTE SIMULATION ---
    # Initialize the simulator with settings
    sim = OfflineSimulator(
        bgc_model_choice=bgc_model_choice, 
        exp_name=exp_name,
        dt_phys=dt_in_sec, 
        sponge_width=sponge_choice,
        tau_lateral=tau_lateral_choice, 
        tau_bottom=tau_bottom_choice, 
        tau_coast=tau_coast_choice, 
        is_global=is_global_choice
    )
    
    # Hand over all raw data and let the simulator subset and prepare it
    sim.prepare_forcing(
        lat_range=lat_range, 
        lon_range=lon_range, 
        depth_range=depth_range, 
        time_range=time_range,
        ds_t=ds_t, 
        ds_s=ds_s, 
        ds_u=ds_u, 
        ds_v=ds_v, 
        ds_sw=ds_sw,
        ds_wind=ds_wind,
        ds_ice=ds_ice, 
        ds_k=ds_k, 
        ds_clim=ds_clim,
        ds_restart=ds_restart,
        glodap_dir=glodap_dir
    )
    
    # Save the driver script to the output directory for reproducibility
    shutil.copy(sys.argv[0], os.path.join(sim.out_dir, os.path.basename(sys.argv[0])))
    
    # Start the simulation loop!
    sim.run()