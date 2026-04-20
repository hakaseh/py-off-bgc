# --- Main execution program of py-off-bgc ---
#
# Modify the contents below based on your setup and run it by `python simulate.py`

import os
# Set this BEFORE importing any other modules
os.environ["NUMBA_NUM_THREADS"] = "16"

import shutil
import sys
import numpy as np
import xarray as xr
from source import bgc_models
from source.simulator import OfflineSimulator

# --- Start of user specification ---

# --- 1. CONFIGURATION ---
# Your output directory will be named {input_name}_{bgc_model_choice}_{extra_name}
input_name = "BRAN2020"
bgc_model_choice = "NPZD" 
extra_name = "fulldepth" # give some unique name
dt_in_sec = 600.0
mld_choice = 0.03 # kg/m3 threshold for surface MLD criterion
sponge_choice = 1
tau_lateral_choice = 86400.0 * 1
tau_bottom_choice = 86400.0 * 30
is_global_choice = True
restart_date = "20160331"
restart_file = f"output/{bgc_model_choice}_{input_name}_{extra_name}/output_{bgc_model_choice}_{input_name}_{extra_name}_{restart_date}.nc"
restart_file = None
clim_file = f"climatology/{input_name}/GLODAPv2.2016b.ALL_{input_name}.nc"

# Domain Slicing
lat_range   = slice(-60, 60) #slice(17, 50)
lon_range   = slice(None, None) #117, 150)
depth_range = slice(None, None) #0, 1000) # 0, 300)
time_range  = slice("20160101","20161231")#None, None) 

# --- 2. LOAD & SLICE DATA ---
# Mandatory input (modify as necessary)
ds_t = xr.open_mfdataset("path_to_file")
ds_s = xr.open_mfdataset("path_to_file")
ds_u = xr.open_mfdataset("path_to_file")
ds_v = xr.open_mfdataset("path_to_file")
ds_sw = xr.open_mfdataset("path_to_file")
ds_wind = xr.open_mfdataset("path_to_file")
# Optional input (set to None if not providing)
ds_k = xr.open_mfdataset("path_to_file")
ds_ice = xr.open_mfdataset("path_to_file")


# --- End of user specification ---
# ---
# You should not have to modify the rest of the code

if file_k:
    print("  -> Found Kz file. Using standard diffusion.")
    mixing_choice = "diffusion" 
else:
    print("  -> No Kz file found!")
    print("     Creating dummy Kz array and forcing 'convective' mixing.")
    # Creates a dummy array of exactly the right shape and coordinates
    ds_k = xr.zeros_like(ds_t) 
    mixing_choice = "convective"

if not ds_ice:
    print("Ice concentration data was not provided so setting all to ONE (no ice cover).")
    ds_ice = xr.ones_like(ds_t)

# Load BGC Inputs exactly the same way!
ds_clim = xr.open_mfdataset(clim_file).sel(depth=depth_range,lat=lat_range,lon=lon_range)

ds_restart = None
if restart_file:
    ds_restart = xr.open_dataset(restart_file).sel(depth=depth_range,lat=lat_range,lon=lon_range)

# Subsetting in space and time
ds_t = ds_t.sel(time=time_range, depth=depth_range, lat=lat_range, lon=lon_range)
ds_s = ds_s.sel(time=time_range, depth=depth_range, lat=lat_range, lon=lon_range)
ds_u = ds_u.sel(time=time_range, depth=depth_range, lat=lat_range, lon=lon_range)
ds_v = ds_v.sel(time=time_range, depth=depth_range, lat=lat_range, lon=lon_range)
ds_k = ds_k.sel(time=time_range, depth=depth_range, lat=lat_range, lon=lon_range)
ds_sw = ds_sw.sel(time=time_range, lat=lat_range, lon=lon_range)
ds_wind = ds_wind.sel(time=time_range, lat=lat_range, lon=lon_range)
ds_ice = ds_ice.sel(time=time_range, lat=lat_range, lon=lon_range)

nz, ny, nx = ds_t.depth.size, ds_t.lat.size, ds_t.lon.size
mask = ~np.isnan(ds_t.isel(time=0).values)


# --- 3. RUN SIMULATION ---
print(f"\nLoading BGC Model: {bgc_model_choice}...")
model = bgc_models.get_model(bgc_model_choice, nz, ny, nx, mask)
# If you want, modify the bgc model parameters here
#model.p_sin = 0.1

# Pass the perfectly sliced datasets
model.initialize(ds_restart=ds_restart, ds_clim=ds_clim)
if restart_file:
    ds_restart.close()

sim = OfflineSimulator(
    da_t = ds_t, 
    bgc_model = model, 
    model_name = bgc_model_choice, 
    exp_name = input_name,
    extra_name = extra_name,
    dt_phys = dt_in_sec,
    ds_clim = ds_clim, 
    mixing_method = mixing_choice,
    mld_threshold = mld_choice,
    sponge_width = sponge_choice, 
    tau_lateral = tau_lateral_choice, 
    tau_bottom = tau_bottom_choice,
    is_global = is_global_choice
)

# Save this file to the output directory for reproducibility
shutil.copy(sys.argv[0], os.path.join(sim.out_dir, "simulate.py"))

# Start the simulation
sim.run(ds_u, ds_v, ds_k, ds_t, ds_s, ds_sw)