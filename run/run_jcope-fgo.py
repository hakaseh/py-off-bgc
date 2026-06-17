import os
import sys
import shutil
import numpy as np
import xarray as xr
from source.simulator import OfflineSimulator

# --- 1. CONFIGURATION ---
exp_name = "JCOPE-FGO"
bgc_model_choice = "NEMURO"
dt_in_sec = 900
mld_choice = 0.03
sponge_choice = 1
tau_lateral_choice = 86400.0 * 1
tau_bottom_choice = 86400.0 * 30
is_global_choice = False
restart_file = None #f"output/{exp_name}/{bgc_model_choice}/restart.nc"
clim_file = f"climatology/{exp_name}/GLODAPv2.2016b.ALL_{exp_name}.nc"

# Domain Slicing
lat_range   = slice(None, None) #17, 50)
lon_range   = slice(None, None) #117, 150)
depth_range = slice(None, None) #0, 300)
time_range  = slice(None, None) #"20160101", "20161231")

# --- 2. LOAD PHYSICAL DATA (CMEMS SPECIFIC) ---
print("Loading raw datasets...")
ds_t = xr.open_mfdataset('input/JCOPE-FGO/interp_JCOPE-FGO_NWP_T.nc*')['tm']
ds_s = xr.open_mfdataset('input/JCOPE-FGO/interp_JCOPE-FGO_NWP_S.nc*')['sm']
ds_u = xr.open_mfdataset('input/JCOPE-FGO/interp_JCOPE-FGO_NWP_U.nc*')['um']
ds_v = xr.open_mfdataset('input/JCOPE-FGO/interp_JCOPE-FGO_NWP_V.nc*')['vm']
ds_sw = xr.open_mfdataset('input/JCOPE-FGO/nc_sigma/NSWR/NSWR_*_NWP.nc').squeeze()['nswrm']

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