import os
import numpy as np
import xarray as xr
from abc import ABC, abstractmethod
from .utils import apply_sinking, GLODAP_MAP

class BaseBGCModel(ABC):
    def __init__(self, nz, ny, nx, water_mask):
        self.nz, self.ny, self.nx = nz, ny, nx
        self.water_mask = water_mask
        self.tracers = {} 
        self.names = []
        self.sinking_config = {}
        
    def initialize(self, ds_restart=None, ds_clim=None):
            """Universal Initialization from prepared xarray Datasets."""
            
            if ds_restart is None and ds_clim is None:
                raise ValueError("Initialization Error: Provide either ds_restart or ds_clim.")
    
            # 1. RESTART (Overrides Climatology)
            if ds_restart is not None:
                print("  [Init] Initializing from Restart Dataset")
                for name in self.names:
                    if name in ds_restart:
                        data = np.nan_to_num(ds_restart[name].values)
                        self.tracers[name][:] = np.where(self.water_mask, data, 0.0)
                    else:
                        print(f"    WARNING: '{name}' missing in restart! Setting to 0.01")
                        self.tracers[name][:] = np.where(self.water_mask, 0.01, 0.0)
            
            # 2. CLIMATOLOGY
            else:
                print("  [Init] Initializing from Climatology Dataset")
                for name in self.names:
                    clim_var = GLODAP_MAP.get(name, name)
                    
                    if clim_var in ds_clim:
                        print(f"    -> Loading '{name}' (from '{clim_var}')")
                        data = np.nan_to_num(ds_clim[clim_var].values)
                        self.tracers[name][:] = np.where(self.water_mask, data, 0.0)
                    else:
                        print(f"    -> '{name}' not in climatology. Seeding with 0.01.")
                        self.tracers[name][:] = np.where(self.water_mask, 0.01, 0.0)

    @abstractmethod
    def biology_step(self, t_curr, sw_curr, dz, dt):
        pass

    def sinking_step(self, dz, dt):
        for name, speed in self.sinking_config.items():
            if name in self.tracers:
                # We use [:] to ensure we are updating the existing array 
                # memory rather than replacing the object reference.
                self.tracers[name][:] = apply_sinking(
                    self.tracers[name], 
                    dz, 
                    dt, 
                    speed
                )