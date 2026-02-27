import numpy as np
import xarray as xr
import bgc_models
from simulator import OfflineSimulator

# --- 1. CONFIGURATION ---
# Your output directory will be named {input_name}_{bgc_model_choice}_{extra_name}
input_name = "LORA"
infile_suffix = "npac.*.nc"
bgc_model_choice = "NPZD" 
extra_name = "diffusion"
dt_in_sec = 1200.0  
mld_choice = 0.03 # kg/m3 threshold for surface MLD criterion
sponge_choice = 1
tau_lateral_choice = 86400.0 * 1
tau_bottom_choice = 86400.0 * 30
restart_date = "20160331"
restart_file = f"../output/{bgc_model_choice}_{input_name}_{extra_name}/output_{bgc_model_choice}_{input_name}_{extra_name}_{restart_date}.nc"
restart_file = None

# Files
dir_exp = f"../input/{input_name}"
file_t = f"{dir_exp}/input_{input_name}_t_{infile_suffix}"
file_s = f"{dir_exp}/input_{input_name}_s_{infile_suffix}"
file_u = f"{dir_exp}/input_{input_name}_u_{infile_suffix}"
file_v = f"{dir_exp}/input_{input_name}_v_{infile_suffix}"
file_k = f"{dir_exp}/input_{input_name}_k_{infile_suffix}"
file_sw = f"{dir_exp}/input_{input_name}_swr_{infile_suffix}"
clim_file = f"../climatology/{input_name}/GLODAPv2.2016b.ALL_{input_name}.nc"
 

# Domain Slicing
lat_range   = slice(None, None) #slice(17, 50)
lon_range   = slice(None, None) #117, 150)
depth_range = slice(None, None) #0, 1000) # 0, 300)
time_range  = slice("20160101","20161231")#None, None) 

# --- 2. LOAD & SLICE DATA ---
print("Loading and slicing datasets...")
ds_t = xr.open_mfdataset(file_t)["t"].sel(lat=lat_range, lon=lon_range, depth=depth_range, time=time_range)
ds_s = xr.open_mfdataset(file_s)["s"].sel(lat=lat_range, lon=lon_range, depth=depth_range, time=time_range)
ds_u = xr.open_mfdataset(file_u)["u"].sel(lat=lat_range, lon=lon_range, depth=depth_range, time=time_range)
ds_v = xr.open_mfdataset(file_v)["v"].sel(lat=lat_range, lon=lon_range, depth=depth_range, time=time_range)
ds_sw = xr.open_mfdataset(file_sw)["swr"].sel(lat=lat_range, lon=lon_range, time=time_range).squeeze()

# --- MISSING K_z LOGIC ---
if file_k:
    print("  -> Found Kz file. Using standard diffusion.")
    ds_k = xr.open_mfdataset(file_k)["k"].sel(lat=lat_range, lon=lon_range, depth=depth_range, time=time_range)
    mixing_choice = "diffusion" 
else:
    print("  -> WARNING: No Kz file found!")
    print("     Creating dummy Kz array and forcing 'convective' mixing.")
    # Creates a dummy array of exactly the right shape and coordinates
    ds_k = xr.zeros_like(ds_t) 
    mixing_choice = "convective"

# Load BGC Inputs exactly the same way!
ds_clim = xr.open_mfdataset(clim_file).sel(lat=lat_range, lon=lon_range, depth=depth_range)

ds_restart = None
if restart_file:
    ds_restart = xr.open_dataset(restart_file).sel(lat=lat_range, lon=lon_range, depth=depth_range)

nz, ny, nx = ds_t.shape[1:]
mask = ~np.isnan(ds_t.isel(time=0).values)


# --- 3. RUN SIMULATION ---
print(f"\nLoading BGC Model: {bgc_model_choice}...")
model = bgc_models.get_model(bgc_model_choice, nz, ny, nx, mask)

# Pass the perfectly sliced datasets
model.initialize(ds_restart=ds_restart, ds_clim=ds_clim)
if restart_file:
    ds_restart.close()

sim = OfflineSimulator(
    da_t = ds_t, 
    model = model, 
    model_name = bgc_model_choice, 
    exp_name = input_name,
    extra_name = extra_name,
    dt_phys = dt_in_sec,
    ds_clim = ds_clim, 
    mixing_method = mixing_choice,
    mld_threshold = mld_choice,
    sponge_width = sponge_choice, 
    tau_lateral = tau_lateral_choice, 
    tau_bottom = tau_bottom_choice
)
sim.run(ds_u, ds_v, ds_k, ds_t, ds_s, ds_sw)