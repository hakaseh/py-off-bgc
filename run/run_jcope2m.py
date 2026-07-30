import os

# Set this to the actual number of cores you want to use
NUM_CORES = "12" 

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
base_dir = "/snow/hakaseh/py-off-bgc"
exp_name = "JCOPE2M"
bgc_model_choice = "NEMURO"
dt_in_sec = 900
sponge_choice = 1
tau_lateral_choice = 86400.0 * 1
tau_bottom_choice = 86400.0 * 30
tau_coast_choice = 86400.0 * 1
is_global_choice = False
restart_file = None #f"output/{exp_name}/{bgc_model_choice}/output_{exp_name}_{bgc_model_choice}_{year-1}1231.nc"
clim_file = None
glodap_dir = f"{base_dir}/climatology/GLODAPv2.2016b.MappedClimatologies/"
year = 2010

# Domain Slicing
lat_range   = slice(17,50)
lon_range   = slice(117,150)
depth_range = slice(None, 1000)
time_range  = slice(f'{year}0101', f'{year}1231')

# A safer preprocess focusing only on spatial reduction
# not doing depth slicing because JCOPE2M depth values are negative. Deal with this issue later...
def make_spatial_prep(lo_range, la_range):
    def ds_prep(ds):
        selectors = {'lat': la_range, 'lon': lo_range}
        #if "depth" in ds.dims:
        #    selectors['depth'] = de_range
        return ds.sel(**selectors)
    return ds_prep

# --- 2. LOAD PHYSICAL DATA (CMEMS SPECIFIC) ---
print("Loading raw datasets...")
ds_t = xr.open_mfdataset(f'{base_dir}/input/JCOPE2M/best_estimate/T_{year}*.nc', preprocess=make_spatial_prep(lon_range, lat_range))['TT']
ds_s = xr.open_mfdataset(f'{base_dir}/input/JCOPE2M/best_estimate/S_{year}*.nc', preprocess=make_spatial_prep(lon_range, lat_range))['ST']
ds_u = xr.open_mfdataset(f'{base_dir}/input/JCOPE2M/best_estimate/UI_{year}*.nc', preprocess=make_spatial_prep(lon_range, lat_range))['UI']
ds_v = xr.open_mfdataset(f'{base_dir}/input/JCOPE2M/best_estimate/VI_{year}*.nc', preprocess=make_spatial_prep(lon_range, lat_range))['VI']
ds_jra = xr.open_mfdataset(f'/home/hakaseh/data/JRA55-do-1-6-0/wget/rsds_input4MIPs_atmosphericState_OMIP_MRI-JRA55-do-1-6-0_gr_{year}*.nc', preprocess=make_spatial_prep(lon_range, lat_range))['rsds']
ds_jra_daily = ds_jra.resample(time='1D').mean()
ds_interpolated = ds_jra_daily.interp(
    lat=ds_t['lat'], 
    lon=ds_t['lon'], 
    method='linear'  # You can also use 'nearest' if you don't want to blend values
)
ds_sw = ds_interpolated.drop_vars(["lat_bnds","lon_bnds","time_bnds"],errors="ignore")

# JCOPE2M provides negative depth values, which we convert to positives
ds_t = ds_t.assign_coords(depth=ds_t['depth'] * -1.0)
ds_s = ds_s.assign_coords(depth=ds_s['depth'] * -1.0)
ds_u = ds_u.assign_coords(depth=ds_u['depth'] * -1.0)
ds_v = ds_v.assign_coords(depth=ds_v['depth'] * -1.0)

# Optional input
ds_k = None
ds_ice = None
ds_jra = np.sqrt( xr.open_mfdataset(f'/home/hakaseh/data/JRA55-do-1-6-0/wget/uas_input4MIPs_atmosphericState_OMIP_MRI-JRA55-do-1-6-0_gr_{year}*.nc', 
                           preprocess=make_spatial_prep(lon_range, lat_range))['uas']**2 + 
                  xr.open_mfdataset(f'/home/hakaseh/data/JRA55-do-1-6-0/wget/vas_input4MIPs_atmosphericState_OMIP_MRI-JRA55-do-1-6-0_gr_{year}*.nc', 
                           preprocess=make_spatial_prep(lon_range, lat_range))['vas']**2 )
ds_jra_daily = ds_jra.resample(time='1D').mean()
ds_interpolated = ds_jra_daily.interp(
    lat=ds_t['lat'], 
    lon=ds_t['lon'], 
    method='linear'  # You can also use 'nearest' if you don't want to blend values
)
ds_wind = ds_interpolated.drop_vars(["lat_bnds","lon_bnds","time_bnds"],errors="ignore")

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
    glodap_dir=glodap_dir,
    ds_restart=ds_restart
)

# Save the driver script to the output directory for reproducibility
shutil.copy(sys.argv[0], os.path.join(sim.out_dir, os.path.basename(sys.argv[0])))

# Start the simulation loop!
sim.run()