import xarray as xr
import numpy as np
import os
import sys
import yaml
import copernicusmarine
import cdsapi
import pandas as pd
import io
import requests


# Main program
def download_input(cfg, run_name):

    ocean_source = cfg["OCEAN_SOURCE"]
    atmos_source = cfg["ATMOS_SOURCE"]
    ref_ocean = cfg["REF_OCEAN"]
    var_ocean = cfg["VAR_OCEAN"]
    lon0, lon1 = cfg['LON_BOUNDS']
    lat0, lat1 = cfg['LAT_BOUNDS']
    dep0, dep1 = cfg['DEPTH_BOUNDS']
    date0, date1 = cfg['DATE_BOUNDS']
    
    lat_range = slice(lat0, lat1)
    lon_range = slice(lon0, lon1)
    depth_range = slice(dep0, dep1)
    time_range = slice(date0, date1)

    if ocean_source is not None:
        # Specify the source data ID and variables
        if ocean_source == "copernicus_glorys12v1":
            id_data = "cmems_mod_glo_phy_my_0.083deg_P1D-m"
            list_3d = ["thetao", "so", "uo", "vo"]
            list_2d = ["siconc"]
        elif ocean_source == "copernicus_goepr":
            id_data = "cmems_mod_glo_phy-mnstd_my_0.25deg_P1D-m"
            list_3d = ["thetao_mean", "so_mean", "uo_mean", "vo_mean"]
            list_2d = ["siconc_mean"]
     
        # Loop over the 3D variables
        for i in range(len(list_3d)):
            fname = f"input/{run_name}/{ocean_source}_{list_3d[i]}_{date0}-{date1}.nc"            
            if not os.path.exists(fname):
                ds = copernicusmarine.open_dataset(
                    dataset_id = id_data, 
                    variables = [list_3d[i]],
                    minimum_longitude = lon0,
                    maximum_longitude = lon1,
                    minimum_latitude = lat0,
                    maximum_latitude = lat1,
                    minimum_depth = dep0,
                    maximum_depth = dep1,
                    start_datetime = date0,
                    end_datetime = date1
                )
                ds = ds.rename({"latitude": "lat", "longitude": "lon"})
                ds.to_netcdf(fname)
                print(f"Saved: {fname}")
            else:
                print(f"{fname} already exists. skipped.")

            if i == 0:
                # store the reference grid
                ds_ref = xr.open_dataset(fname)

        # Loop over the 2D variables (e.g., sea ice concentration)
        for i in range(len(list_2d)):
            if not os.path.exists(fname):
                fname = f"input/{run_name}/{ocean_source}_{list_2d[i]}_{date0}-{date1}.nc"
                ds = copernicusmarine.open_dataset(
                    dataset_id = id_data, 
                    variables = [list_2d[i]],
                    minimum_longitude = lon0,
                    maximum_longitude = lon1,
                    minimum_latitude = lat0,
                    maximum_latitude = lat1,
                    start_datetime = date0,
                    end_datetime = date1
                )
                ds = ds.rename({"latitude": "lat", "longitude": "lon"})
                ds.to_netcdf(fname)
                print(f"Saved: {fname}")
            else:
                print(f"{fname} already exists. skipped.")                

    # Use the existing ocean data for atmos interpolation
    else:
        # load the reference ocean input data (to which the atmos data are interpolated)
        ds_ref = xr.open_dataset(f"input/{run_name}/{ref_ocean}")
        ds_ref = rename_ds(ds_ref)

    if atmos_source is not None:
            
        if atmos_source == "cds_era5":
            final_fname = f"input/{run_name}/{atmos_source}_{date0}-{date1}.nc"
            
            # 1. Automatically generate a list of months covering your date range
            periods = pd.period_range(start=date0, end=date1, freq='M')
            
            c = cdsapi.Client()
            north, south = max(lat0, lat1), min(lat0, lat1)
            west, east = min(lon0, lon1), max(lon0, lon1)
            
            raw_files = []
            
            # 2. Loop through each month and download wind and solar separately
            for p in periods:
                m_start = max(pd.to_datetime(date0), p.start_time).strftime('%Y-%m-%d')
                m_end = min(pd.to_datetime(date1), p.end_time).strftime('%Y-%m-%d')
                
                raw_wind_fname = f"input/{run_name}/{atmos_source}_wind_{m_start}_to_{m_end}.nc"
                raw_solar_fname = f"input/{run_name}/{atmos_source}_solar_{m_start}_to_{m_end}.nc"
                                
                # --- DOWNLOAD WIND (Instantaneous) ---
                if not os.path.exists(raw_wind_fname):
                    print(f"Downloading ERA5 Wind for {m_start} to {m_end}...")
                    # 1. Send the request WITHOUT a target filename
                    result = c.retrieve(
                        'reanalysis-era5-single-levels',
                        {
                            'product_type': 'reanalysis',
                            'variable': [
                                '10m_u_component_of_wind', 
                                '10m_v_component_of_wind', 
                            ],
                            'date': f"{m_start}/{m_end}",
                            'time': [f"{str(h).zfill(2)}:00" for h in range(24)],
                            'area': [north, west, south, east],
                            'format': 'netcdf',
                        }
                    )

                    # 2. Fetch the data payload from the provided URL directly into RAM
                    response = requests.get(result.location)

                    # 3. Load the in-memory bytes into an Xarray Dataset
                    ds_wind = xr.open_dataset(io.BytesIO(response.content), engine='h5netcdf')
                    
                    ds_wind = rename_ds(ds_wind)
                    ds_wind = resample_regrid(ds_wind, ds_ref)
                    ds_wind.to_netcdf(raw_wind_fname)
                else:
                    print(f"Raw wind file {raw_wind_fname} already exists. Skipping.")

                # --- DOWNLOAD SOLAR (Averaged Flux) ---
                if not os.path.exists(raw_solar_fname):
                    print(f"Downloading ERA5 Solar for {m_start} to {m_end}...")
                    # 1. Send the request WITHOUT a target filename
                    result = c.retrieve(
                        'reanalysis-era5-single-levels',
                        {
                            'product_type': 'reanalysis',
                            'variable': [
                                'mean_surface_downward_short_wave_radiation_flux'
                            ],
                            'date': f"{m_start}/{m_end}",
                            'time': [f"{str(h).zfill(2)}:00" for h in range(24)],
                            'area': [north, west, south, east],
                            'format': 'netcdf',
                        }
                    )

                    # 2. Fetch the data payload from the provided URL directly into RAM
                    response = requests.get(result.location)

                    # 3. Load the in-memory bytes into an Xarray Dataset
                    ds_solar = xr.open_dataset(io.BytesIO(response.content), engine='h5netcdf')

                    ds_solar = rename_ds(ds_solar)
                    ds_solar = resample_regrid(ds_solar, ds_ref)
                    ds_solar.to_netcdf(raw_solar_fname)
                else:
                    print(f"Raw solar file {raw_solar_fname} already exists. Skipping.")
                                   

# Rename from raw files
def rename_ds(ds):    
    rename_map = {"longitude": "lon", 
                  "latitude": "lat", 
                  "valid_time": "time"
                 }
    rename_dict = {k: v for k, v in rename_map.items() if k in ds.dims}
    ds = ds.rename(rename_dict)
    return ds

# Resample/Regrid
def resample_regrid(ds,ds_ref):
    ds = ds.resample(time="1D").mean()
    ds = ds.interp_like(ds_ref.isel(depth=0,time=0).squeeze())
    return ds


# ==========================================
# COMMAND LINE INTERFACE
# ==========================================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python prep_atmos.py <path_to_config.yaml>")
        sys.exit(1)

    config_file_path = sys.argv[1]

    if not os.path.exists(config_file_path):
        print(f"Error: Could not find configuration file at '{config_file_path}'")
        sys.exit(1)

    with open(config_file_path, 'r') as file:
        cfg = yaml.safe_load(file)

    # --- NEW LOGIC HERE ---
    # 1. os.path.basename removes "configs/" -> "hokkaido_era5.yml"
    # 2. os.path.splitext removes ".yml" -> "hokkaido_era5"
    run_name = os.path.splitext(os.path.basename(config_file_path))[0]
    # Create a directory inside "input" if it does not exist
    os.makedirs(f"input/{run_name}", exist_ok=True)

    # Run the program
    download_input(cfg, run_name)