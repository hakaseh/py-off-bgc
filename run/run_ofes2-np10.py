import os
import sys
import shutil
import numpy as np
import xarray as xr
from source.simulator import OfflineSimulator
from source.bgc_models.npzd_sasai import Params_BGC

# --- 1. CONFIGURATION ---
exp_name = "OFES2-NP10"
bgc_model_choice = "NPZD"
dt_in_sec = 600.0
mld_choice = 0.03 
sponge_choice = 1
tau_lateral_choice = 86400.0 * 1
tau_bottom_choice = 86400.0 * 30
is_global_choice = False
restart_file = f"input/{exp_name}/restart_{exp_name}_20180101.nc"
clim_file = None #f"climatology/{exp_name}/GLODAPv2.2016b.ALL_{exp_name}.nc"

# Domain Slicing
lat_range   = slice(None, None) #17, 50)
lon_range   = slice(None, None) #117, 150)
depth_range = slice(None, None)
time_range  = slice(None, None) #"20160101", "20161231")

# --- 2. LOAD PHYSICAL DATA (CMEMS SPECIFIC) ---
print("Loading raw datasets...")
ds_t = xr.open_dataset(f'input/{exp_name}/temp.nc')['temp']
ds_s = xr.open_dataset(f'input/{exp_name}/salinity.nc')['salinity']
# divide by 100 to convert from cm/s to m/s
ds_u = 1e-2 * xr.open_dataset(f'input/{exp_name}/u.nc')['u']
ds_v = 1e-2 * xr.open_dataset(f'input/{exp_name}/v.nc')['v']
ds_sw = xr.open_dataset(f'input/{exp_name}/rsds_JRA55-do-1-6-0_OFES2_NP10_2018.nc')['rsds']

# Optional input
ds_wind = None
ds_k = None
ds_ice = None

# BGC Inputs
ds_clim = xr.open_mfdataset(clim_file) if clim_file else None
ds_restart = xr.open_dataset(restart_file) if restart_file else None

# Rename to required dimensions
ds_t = ds_t.rename({"lev": "depth"})
ds_s = ds_s.rename({"lev": "depth"})
ds_u = ds_u.rename({"lev": "depth"})
ds_v = ds_v.rename({"lev": "depth"})
#ds_sw = ds_sw.rename({"longitude": "lon", "latitude": "lat", "valid_time": "time"})
#ds_wind = ds_wind.rename({"longitude": "lon", "latitude": "lat", "valid_time": "time"})

# Interpolate if necessary
#ds_sw = ds_sw.resample(time="1D").mean()
#ds_wind = ds_wind.resample(time="1D").mean()

# Create the parameter object
custom_bgc = Params_BGC()

# Declare your custom values! (Everything else stays as default)
custom_bgc.p_res = 0 
custom_bgc.p_exc = 0     
custom_bgc.p_gro = 1
custom_bgc.p_gef = 0.7
custom_bgc.p_zmo = 0.24
custom_bgc.p_pmo = 0.12

# --- 3. EXECUTE SIMULATION ---
# Initialize the simulator with settings
sim = OfflineSimulator(
    bgc_model_choice=bgc_model_choice, exp_name=exp_name,
    dt_phys=dt_in_sec, 
    bgc_params=custom_bgc,
    mld_threshold=mld_choice, sponge_width=sponge_choice,
    tau_lateral=tau_lateral_choice, tau_bottom=tau_bottom_choice, is_global=is_global_choice
)

# Hand over all raw data and let the simulator subset and prepare it
sim.prepare_forcing(
    lat_range=lat_range, lon_range=lon_range, depth_range=depth_range, time_range=time_range,
    ds_t=ds_t, ds_s=ds_s, ds_u=ds_u, ds_v=ds_v, ds_sw=ds_sw, ds_wind=ds_wind,
    ds_ice=ds_ice, ds_k=ds_k, ds_clim=ds_clim, ds_restart=ds_restart
)

# Save the driver script to the output directory for reproducibility
shutil.copy(sys.argv[0], os.path.join(sim.out_dir, os.path.basename(sys.argv[0])))

# Start the simulation loop!
sim.run()