import os

# Set this to the actual number of cores you want to use
NUM_CORES = "8" 

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
from source.copernicus_data import *

# --- 1. CONFIGURATION ---
exp_name = "COPERNICUS"
bgc_model_choice = "NEMURO"
dt_in_sec = 300
mld_choice = 0.03 
sponge_choice = 1
tau_lateral_choice = 86400.0 * 1
tau_bottom_choice = 86400.0 * 30
tau_coast_choice = 86400.0 * 1
is_global_choice = False
restart_file = None #f"output/{exp_name}/{bgc_model_choice}/output_{exp_name}_{bgc_model_choice}_20191231.nc"
clim_file = None #f"climatology/{exp_name}/GLODAPv2.2016b.ALL_{exp_name}.nc"
glodap_dir = "climatology/GLODAPv2.2016b.MappedClimatologies/"

lon0, lon1 = 117, 150
lat0, lat1 = 17, 50
dep0, dep1 = None, 1000
date0, date1 = "20230101", "20231231"

# Domain Slicing
lat_range   = slice(lat0,lat1)
lon_range   = slice(lon0,lon1)
depth_range = slice(dep0,dep1)
time_range  = slice(date0,date1)

# --- 2. LOAD PHYSICAL DATA (conversion and interpolation are not needed, as done during pre-processing) ---
print("Loading raw datasets...")
ds_t = load_copernicus_ocean("cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m", "thetao", lon0, lon1, lat0, lat1, dep0, dep1, date0, date1)
ds_s = load_copernicus_ocean("cmems_mod_glo_phy-so_anfc_0.083deg_P1D-m", "so", lon0, lon1, lat0, lat1, dep0, dep1, date0, date1)
ds_u = load_copernicus_ocean("cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m", "uo", lon0, lon1, lat0, lat1, dep0, dep1, date0, date1)
ds_v = load_copernicus_ocean("cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m", "vo", lon0, lon1, lat0, lat1, dep0, dep1, date0, date1)
ds_sw, ds_wind = load_era5_sw_and_wind(ds_t, lon0, lon1, lat0, lat1, date0, date1)
ds_k = None
ds_ice = load_copernicus_ocean("cmems_mod_glo_phy_anfc_0.083deg_P1D-m", "siconc", lon0, lon1, lat0, lat1, dep0, dep1, date0, date1)

# BGC Inputs
ds_clim = xr.open_mfdataset(clim_file) if clim_file else None
ds_restart = xr.open_dataset(restart_file) if restart_file else None

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