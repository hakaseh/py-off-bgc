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

# --- 1. CONFIGURATION ---
exp_name = "LORA-WNP"
bgc_model_choice = "FENNEL06"
dt_in_sec = 1200
mld_choice = 0.03 
sponge_choice = 1
tau_lateral_choice = 86400.0 * 1
tau_bottom_choice = 86400.0 * 30
tau_coast_choice = 86400.0 * 1
is_global_choice = False
restart_file = None #f"output/{bgc_model_choice}_{exp_name}/restart.nc"
clim_file = None #f"climatology/{exp_name}/GLODAPv2.2016b.ALL_{exp_name}.nc"
glodap_dir = "climatology/GLODAPv2.2016b.MappedClimatologies/"

# Domain Slicing
lat_range   = slice(17,50)
lon_range   = slice(117,150)
depth_range = slice(None, None)
time_range  = slice('20160101', '20231231')

# --- 2. LOAD PHYSICAL DATA (CMEMS SPECIFIC) ---
print("Loading raw datasets...")
ds_t = xr.open_mfdataset('input/LORA-WNP/input_LORA_t_npac.20*.nc')['t']
ds_s = xr.open_mfdataset('input/LORA-WNP/input_LORA_s_npac.20*.nc')['s']
ds_u = xr.open_mfdataset('input/LORA-WNP/input_LORA_u_npac.20*.nc')['u']
ds_v = xr.open_mfdataset('input/LORA-WNP/input_LORA_v_npac.20*.nc')['v']
ds_sw = xr.open_mfdataset('input/LORA-WNP/input_LORA_swr_npac.20*.nc')['swr']

# Optional input
ds_k = None
ds_ice = None
ds_wind = None

# BGC Inputs
ds_clim = xr.open_mfdataset(clim_file) if clim_file else None
ds_restart = xr.open_dataset(restart_file) if restart_file else None

# Rename to required dimensions (lon, lat, depth, time)
#ds_t = ds_t.rename({"longitude": "lon", "latitude": "lat"})

# Interpolate if necessary
#ds_sw = ds_sw.resample(time="1D").mean()
#ds_wind = ds_wind.resample(time="1D").mean()

# --- 3. EXECUTE SIMULATION ---
# Initialize the simulator with settings
sim = OfflineSimulator(
    bgc_model_choice=bgc_model_choice, 
    exp_name=exp_name,
    dt_phys=dt_in_sec, 
    mld_threshold=mld_choice, 
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
    glodap_dir=glodap_dir,
    ds_restart=ds_restart
)

# Save the driver script to the output directory for reproducibility
shutil.copy(sys.argv[0], os.path.join(sim.out_dir, os.path.basename(sys.argv[0])))

# Start the simulation loop!
sim.run()