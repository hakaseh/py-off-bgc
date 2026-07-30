import os

# Set this to the actual number of cores you want to use
NUM_CORES = "24" 

# 1. Force XLA's Eigen compiler to use all cores on Linux
os.environ["XLA_FLAGS"] = (
    f"--xla_cpu_multi_thread_eigen=true "
    f"intra_op_parallelism_threads={NUM_CORES} "
    f"inter_op_parallelism_threads={NUM_CORES}"
)

# 2. Force the underlying C++ math libraries to match
os.environ["OMP_NUM_THREADS"] = NUM_CORES
os.environ["OPENBLAS_NUM_THREADS"] = NUM_CORES
os.environ["MKL_NUM_THREADS"] = NUM_CORES

import sys
import shutil
import numpy as np
import xarray as xr
from source.simulator import OfflineSimulator

# --- 1. CONFIGURATION ---
exp_name = "FORA-NP60"
bgc_model_choice = "NEMURO"
dt_in_sec = 900
sponge_choice = 1
tau_lateral_choice = 86400.0 * 1
tau_bottom_choice = 86400.0 * 30
tau_coast_choice = 86400.0 * 1
is_global_choice = False
restart_file = None #f"output/{exp_name}/{bgc_model_choice}/output_{exp_name}_{bgc_model_choice}_19930519.nc"
clim_file = None #f"climatology/{exp_name}/GLODAPv2.2016b.ALL_{exp_name}.nc"
glodap_dir = "climatology/GLODAPv2.2016b.MappedClimatologies/"

# Domain Slicing
lat_range   = slice(17,50)
lon_range   = slice(117,150)
depth_range = slice(None, 6000)
time_range  = slice(None, None)

# --- 2. LOAD PHYSICAL DATA (conversion and interpolation are not needed, as done during pre-processing) ---
print("Loading raw datasets...")
year = 1993
ds_t = xr.open_mfdataset(f'input/FORA-NP60/t_{year}_nwp.nc')['thetao']
ds_s = xr.open_mfdataset(f'input/FORA-NP60/s_{year}_nwp.nc')['so']
ds_u = xr.open_mfdataset(f'input/FORA-NP60/u_{year}_nwp.nc')['uo']
ds_v = xr.open_mfdataset(f'input/FORA-NP60/v_{year}_nwp.nc')['vo']
ds_sw = xr.open_mfdataset(f'input/FORA-NP60/h_short_{year}_nwp.nc')['rsntds']

# Optional input
ds_k = None #xr.open_mfdataset(f'input/FORA-NP60/avd_{year}_nwp.nc')['difvtro']
ds_ice = xr.open_mfdataset(f'input/FORA-NP60/ice_state_{year}_nwp.nc')['siconc']

#ds_wind_u = xr.open_mfdataset('/mnt/FORA-JPN60/hst_day-glb/2020/nc_sfc_geostr_u2.2020*')['sfcu']
#ds_wind_u = ds_wind_u.interp_like(ds_t)
#ds_wind_v = xr.open_mfdataset('/mnt/FORA-JPN60/hst_day-glb/2020/nc_sfc_geostr_v2.2020*')['sfcv']
#ds_wind_v = ds_wind_v.interp_like(ds_t)
ds_wind = None #np.sqrt(ds_wind_u ** 2 + ds_wind_v **2)

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