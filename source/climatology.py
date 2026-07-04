# source/climatology.py

# Creation of the BGC climatology used for initialization and/or boundary conditions as well as restoring
# * interpolates the GLODAPv2 data to the ocean physics input data
# * for some variables, negative values are set to zeros.
# * for variables expressed in per kg are expressed in per m3 to match with the model units.

import xarray as xr
import numpy as np
import gsw

def standardize_glodap_grid(ds, ref_grid):
    """Aligns GLODAP coordinates and dimensions with the model grid."""
    if "depth_surface" in ds.dims:
        ds = ds.swap_dims({"depth_surface": "Depth"}).rename({"Depth": "depth"})
    
    # Adjust Longitude convention
    if ref_grid['lon'].min() < 0:
        ds.coords['lon'] = (ds.coords['lon'] + 180) % 360 - 180 
    else:
        ds.coords['lon'] = ds.coords['lon'] % 360
        
    return ds.sortby("lon")

def generate_restoring_climatology(ref_da, glodap_dir):
    """
    Generates a grid-aligned BGC climatology dataset from raw GLODAP files.
    Calculates in-situ density for unit conversions and handles NaNs.
    """
    glodap_vars = [
        "Cant", "NO3", "OmegaA", "OmegaC", "oxygen", 
        "pHts25p0", "pHtsinsitutp", "PI_TCO2", "PO4", 
        "salinity", "silicate", "TAlk", "TCO2", "temperature"
    ]
    
    vars_to_convert = [
        "Cant", "NO3", "oxygen", "PI_TCO2", "PO4", 
        "silicate", "TAlk", "TCO2"
    ]
    
    vars_strictly_positive = [
        "Cant", "NO3", "OmegaA", "OmegaC", "oxygen", 
        "PI_TCO2", "PO4", "salinity", "silicate", "TAlk", "TCO2"
    ]

    print("Calculating in-situ density from GLODAP Temperature and Salinity...")
    ds_temp = xr.open_dataset(f"{glodap_dir}/GLODAPv2.2016b.temperature.nc")
    ds_salt = xr.open_dataset(f"{glodap_dir}/GLODAPv2.2016b.salinity.nc")

    ds_temp_std = standardize_glodap_grid(ds_temp, ref_da)["temperature"]
    ds_salt_std = standardize_glodap_grid(ds_salt, ref_da)["salinity"]

    # TEOS-10 Density calculation
    p = ds_temp_std.depth 
    SA = gsw.SA_from_SP(ds_salt_std, p, ds_salt_std.lon, ds_salt_std.lat)
    CT = gsw.CT_from_t(SA, ds_temp_std, p)
    density = gsw.rho(SA, CT, p)

    ds_glodap = xr.Dataset()

    for var in glodap_vars:
        print(f"  -> Processing {var}...")
        ds_var = xr.open_dataset(f"{glodap_dir}/GLODAPv2.2016b.{var}.nc")
        da = standardize_glodap_grid(ds_var, ref_da)[var]
        
        # Density Conversion (umol/kg -> mmol/m^3)
        if var in vars_to_convert:
            da = da * (density / 1000.0)
            da.attrs['units'] = 'mmol/m^3'
        
        # Clip negative artifacts from raw data
        if var in vars_strictly_positive:
            da = da.clip(min=0.0)
            
        ds_glodap[var] = da

    print("Extrapolating NaNs (Flooding)...")
    ds_filled = ds_glodap.interpolate_na(dim='lon', method='nearest', fill_value="extrapolate")
    ds_filled = ds_filled.interpolate_na(dim='lat', method='nearest', fill_value="extrapolate")

    print("Interpolating to Model Grid...")
    ds_regridded = ds_filled.interp_like(ref_da)

    print("Filling vertical gaps...")
    ds_regridded = ds_regridded.ffill(dim='depth')

    for var in vars_strictly_positive:
        ds_regridded[var] = ds_regridded[var].clip(min=0.0)

    print("Applying Target Land Mask...")
    target_mask = ref_da.notnull() 
    
    # This will now correctly punch holes in the BGC data where JCOPE land exists
    ds_final = ds_regridded.where(target_mask)
    
    return ds_final