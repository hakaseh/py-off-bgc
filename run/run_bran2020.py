import os
import sys
import shutil
import numpy as np
import xarray as xr
from source.simulator import OfflineSimulator

# --- 1. CONFIGURATION ---
exp_name = "BRAN2020"
bgc_model_choice = "NEMURO"
dt_in_sec = 1200
mld_choice = 0.03 
sponge_choice = 1
tau_lateral_choice = 86400.0 * 1
tau_bottom_choice = 86400.0 * 30
is_global_choice = False
restart_file = None #f"output/{bgc_model_choice}_{exp_name}/restart.nc"
clim_file = f"climatology/{exp_name}/GLODAPv2.2016b.ALL_{exp_name}.nc"

# Domain Slicing
lat_range   = slice(17, 50)
lon_range   = slice(117, 150)
depth_range = slice(None, 1000) #0, 300)
time_range  = slice("20230101", "20231231")

# --- 2. LOAD PHYSICAL DATA (CMEMS SPECIFIC) ---
print("Loading raw datasets...")
ds_t = xr.open_mfdataset(f'input/{exp_name}/ocean_temp_*.nc', chunks={"time": 1},
                         drop_variables=["Time_bounds", "average_DT"])["temp"].rename(
    {"xt_ocean":"lon","yt_ocean":"lat","st_ocean":"depth","Time":"time"})
ds_s = xr.open_mfdataset(f'input/{exp_name}/ocean_salt_*.nc', chunks={"time": 1},
                         drop_variables=["Time_bounds", "average_DT"])["salt"].rename(
    {"xt_ocean":"lon","yt_ocean":"lat","st_ocean":"depth","Time":"time"})
ds_u = xr.open_mfdataset(f'input/{exp_name}/ocean_u_*.nc', chunks={"time": 1},
                         drop_variables=["Time_bounds", "average_DT"])["u"].rename(
    {"xu_ocean":"lon","yu_ocean":"lat","st_ocean":"depth","Time":"time"})
ds_u = ds_u.interp(lon=ds_s.lon, lat=ds_s.lat, kwargs={"fill_value": "extrapolate"})
ds_v = xr.open_mfdataset(f'input/{exp_name}/ocean_v_*.nc', chunks={"time": 1},
                         drop_variables=["Time_bounds", "average_DT"])["v"].rename(
    {"xu_ocean":"lon","yu_ocean":"lat","st_ocean":"depth","Time":"time"})
ds_v = ds_v.interp(lon=ds_s.lon, lat=ds_s.lat, kwargs={"fill_value": "extrapolate"})
ds_sw = xr.open_mfdataset(f'input/{exp_name}/rsds_*.nc', chunks={"time": 1})["rsds"].rename({"xt_ocean":"lon","yt_ocean":"lat"})

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
    bgc_model_choice=bgc_model_choice, exp_name=exp_name,
    dt_phys=dt_in_sec, mld_threshold=mld_choice, sponge_width=sponge_choice,
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