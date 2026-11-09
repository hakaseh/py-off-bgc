import xarray as xr

def dims_ds(ds, cfg):    
    # Extract bounds from config
    lon0, lon1 = cfg['LON_BOUNDS']
    lat0, lat1 = cfg['LAT_BOUNDS']
    dep0, dep1 = cfg['DEPTH_BOUNDS']
    date0, date1 = cfg['DATE_BOUNDS']

    # Rename dims/coords
    rename_map = {
        "longitude": "lon", 
        "latitude": "lat", 
        "valid_time": "time",
        "dep": "depth"
    }

    # Check both dims and coords to catch all cases
    rename_dict = {k: v for k, v in rename_map.items() if k in ds.dims or k in ds.coords}
    ds = ds.rename(rename_dict)
    
    # Fix negative depth (e.g., jcope2m)
    if "depth" in ds.coords:
        ds = ds.assign_coords(depth=abs(ds["depth"]))

    # Wraps the longitude of an Xarray dataset to match the user's requested 
    # bounding box convention ([0, 360] or [-180, 180]).
    if lon1 > 180:
        print(f"  -> Wrapping lon to [0, 360] convention...")
        ds.coords["lon"] = ds.coords["lon"] % 360
        ds = ds.sortby("lon")
        
    elif lon0 < 0:
        print(f"  -> Wrapping lon to [-180, 180] convention...")
        ds.coords["lon"] = (ds.coords["lon"] + 180) % 360 - 180
        ds = ds.sortby("lon")

    # Sort spatial coordinates to ensure slice(min, max) never returns empty
    sort_dims = [d for d in ["lon", "lat", "depth"] if d in ds.dims]
    if sort_dims:
        ds = ds.sortby(sort_dims)

    return ds

def slice_ds(ds, cfg):    
    # Extract bounds from config
    lon0, lon1 = cfg['LON_BOUNDS']
    lat0, lat1 = cfg['LAT_BOUNDS']
    dep0, dep1 = cfg['DEPTH_BOUNDS']
    date0, date1 = cfg['DATE_BOUNDS']
    
    # Build dynamic selection dictionary
    sel_kwargs = {}
    if "lon" in ds.dims:
        sel_kwargs["lon"] = slice(lon0, lon1)
    if "lat" in ds.dims:
        sel_kwargs["lat"] = slice(lat0, lat1)
    if "depth" in ds.dims:
        sel_kwargs["depth"] = slice(dep0, dep1)
    if "time" in ds.dims:
        sel_kwargs["time"] = slice(date0, date1)
        
    # Apply subset
    if sel_kwargs:
        ds = ds.sel(**sel_kwargs)
        
    return ds