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

# Files
dir_exp = f"input/{input_name}"
file_t = f"{dir_exp}/ocean_temp_*.nc"
file_s = f"{dir_exp}/ocean_salt_*.nc"
file_u = f"{dir_exp}/ocean_u_*.nc"
file_v = f"{dir_exp}/ocean_v_*.nc"
file_k = None #f"{dir_exp}/input_{input_name}_k_{infile_suffix}"
file_sw = f"{dir_exp}/rsds_*.nc"
clim_file = f"climatology/{input_name}/GLODAPv2.2016b.ALL_{input_name}.nc"

# Domain Slicing
lat_range   = slice(-60, 60) #slice(17, 50)
lon_range   = slice(None, None) #117, 150)
depth_range = slice(None, None) #0, 1000) # 0, 300)
time_range  = slice("20160101","20161231")#None, None) 

# --- 2. LOAD & SLICE DATA ---
# CHECK: variable and dimension names (modify if necessary)
print("Loading and slicing datasets...")
ds_t = xr.open_mfdataset(file_t, chunks={"time": 1},
                         drop_variables=["Time_bounds", "average_DT"])["temp"].rename(
    {"xt_ocean":"lon","yt_ocean":"lat","st_ocean":"depth","Time":"time"}).sel(
    time=time_range, depth=depth_range, lat=lat_range, lon=lon_range)
ds_s = xr.open_mfdataset(file_s, chunks={"time": 1},
                         drop_variables=["Time_bounds", "average_DT"])["salt"].rename(
    {"xt_ocean":"lon","yt_ocean":"lat","st_ocean":"depth","Time":"time"}).sel(
    time=time_range, depth=depth_range, lat=lat_range, lon=lon_range)
ds_u = xr.open_mfdataset(file_u, chunks={"time": 1},
                         drop_variables=["Time_bounds", "average_DT"])["u"].rename(
    {"xu_ocean":"lon","yu_ocean":"lat","st_ocean":"depth","Time":"time"}).sel(
    time=time_range, depth=depth_range, lat=lat_range, lon=lon_range)
ds_u = ds_u.interp(lon=ds_s.lon, lat=ds_s.lat, kwargs={"fill_value": "extrapolate"})
ds_v = xr.open_mfdataset(file_v, chunks={"time": 1},
                         drop_variables=["Time_bounds", "average_DT"])["v"].rename(
    {"xu_ocean":"lon","yu_ocean":"lat","st_ocean":"depth","Time":"time"}).sel(
    time=time_range, depth=depth_range, lat=lat_range, lon=lon_range)
ds_v = ds_v.interp(lon=ds_s.lon, lat=ds_s.lat, kwargs={"fill_value": "extrapolate"})
ds_sw = xr.open_mfdataset(file_sw, chunks={"time": 1})["rsds"].rename(
    {"xt_ocean":"lon","yt_ocean":"lat"}).sel(
    time=time_range, lat=lat_range, lon=lon_range)

# --- MISSING K_z LOGIC ---
if file_k:
    print("  -> Found Kz file. Using standard diffusion.")
    ds_k = xr.open_mfdataset(file_k)[name_k].sel(
        time=time_range, depth=depth_range, lat=lat_range, lon=lon_range)
    mixing_choice = "diffusion" 
else:
    print("  -> WARNING: No Kz file found!")
    print("     Creating dummy Kz array and forcing 'convective' mixing.")
    # Creates a dummy array of exactly the right shape and coordinates
    ds_k = xr.zeros_like(ds_t) 
    mixing_choice = "convective"

# Load BGC Inputs exactly the same way!
ds_clim = xr.open_mfdataset(clim_file).sel(depth=depth_range,lat=lat_range,lon=lon_range)

ds_restart = None
if restart_file:
    ds_restart = xr.open_dataset(restart_file).sel(depth=depth_range,lat=lat_range,lon=lon_range)

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