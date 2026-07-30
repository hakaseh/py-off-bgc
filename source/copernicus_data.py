import copernicusmarine
import xarray as xr
import numpy as np

def load_copernicus_ocean(dataset0, variable0, lon0, lon1, lat0, lat1, dep0, dep1, date0, date1):
    
    # 1. Open the lazy dataset
    if variable0 == "siconc": # sea ice fraction, exclude depth
        ds = copernicusmarine.open_dataset(
            dataset_id = dataset0, 
            variables = [variable0],
            minimum_longitude = lon0,
            maximum_longitude = lon1,
            minimum_latitude = lat0,
            maximum_latitude = lat1,
            start_datetime = date0,
            end_datetime = date1
        )        
    else:
        ds = copernicusmarine.open_dataset(
            dataset_id = dataset0, 
            variables = [variable0],
            minimum_longitude = lon0,
            maximum_longitude = lon1,
            minimum_latitude = lat0,
            maximum_latitude = lat1,
            minimum_depth = dep0,
            maximum_depth = dep1,
            start_datetime = date0,
            end_datetime = date1
        )
    
    # 2. Rename the coordinates while the data is still lazy
    # The dictionary maps {"old_name": "new_name"}
    ds = ds.rename({"latitude": "lat", "longitude": "lon"})
    
    return ds[variable0]


def load_era5_sw_and_wind(ds_ref,lon0,lon1,lat0,lat1,date0,date1):

    # 1. Connect to the Google Cloud Zarr store
    ds_era5 = xr.open_zarr(
        "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3", 
        storage_options={"token": "anon"}
    )
    
    # 2. Slice and load the HOURLY data into RAM
    ds_subset = ds_era5[["surface_solar_radiation_downwards", "10m_u_component_of_wind", "10m_v_component_of_wind"]].sel(
        latitude=slice(lat1, lat0), 
        longitude=slice(lon0, lon1), 
        time=slice(date0, date1)
    )
    
    # 3. Calculate the HOURLY wind magnitude 
    # Math: sqrt(u^2 + v^2)
    hourly_wind_speed = np.sqrt(
        ds_subset["10m_u_component_of_wind"]**2 + ds_subset["10m_v_component_of_wind"]**2
    )
    
    # 4. Resample to Daily Mean Wind Speed
    ds_wind = hourly_wind_speed.resample(time="1D").mean().rename("u10_mag")
    
    # 3. Convert units from Accumulated Joules to Continuous Watts/m^2
    # Since ERA5 accumulates this variable over 1 hour (3600 seconds), 
    # dividing by 3600 gives you the average W/m^2 for that hour.
    ds_rad_w_m2 = ds_subset["surface_solar_radiation_downwards"] / 3600.0
    
    # 4. Resample to Daily Means
    # "1D" stands for 1 Day. We tell Xarray to group the hours by day and take the .mean()
    ds_sw = ds_rad_w_m2.resample(time="1D").mean()

    ds_wind = ds_wind.rename({"latitude": "lat", "longitude": "lon"})
    ds_sw = ds_sw.rename({"latitude": "lat", "longitude": "lon"})
    
    # 5. interpolate to the reference grid
    # it is assumed that ds_ref has depth and time, which are removed.
    ds_wind = ds_wind.interp_like(ds_ref.isel(depth=0,time=0).squeeze())
    ds_sw = ds_sw.interp_like(ds_ref.isel(depth=0,time=0).squeeze())

    return ds_sw, ds_wind