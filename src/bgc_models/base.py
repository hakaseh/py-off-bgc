import os
import numpy as np
import xarray as xr
from abc import ABC, abstractmethod
from .utils import apply_sinking_ti, GLODAP_MAP

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
                    self.tracers[name] = np.where(self.water_mask, data, 0.0).astype(np.float32)
                else:
                    print(f"    WARNING: '{name}' missing in restart! Setting to 0.01")
                    self.tracers[name] = np.where(self.water_mask, 0.01, 0.0).astype(np.float32)
                    
        # 2. CLIMATOLOGY
        else:
            print("  [Init] Initializing from Climatology Dataset")
            for name in self.names:
                clim_var = GLODAP_MAP.get(name, name)
                
                if clim_var in ds_clim:
                    print(f"    -> Loading '{name}' (from '{clim_var}')")
                    data = np.nan_to_num(ds_clim[clim_var].values)
                    self.tracers[name] = np.where(self.water_mask, data, 0.0).astype(np.float32)
                else:
                    print(f"    -> '{name}' not in climatology. Seeding with 0.01.")
                    self.tracers[name] = np.where(self.water_mask, 0.01, 0.0).astype(np.float32)

    @abstractmethod
    def biology_step(self, t_curr, sw_curr, dz, dt):
        pass

    def sinking_step(self, dz, dt):
        """Applies sinking to specific tracers defined in sinking_config using Taichi."""
        dz_safe_min = np.where(dz > 1e-6, dz, 1e6)
        min_dz = float(np.min(dz_safe_min))
        
        for name, speed_day in self.sinking_config.items():
            if name in self.tracers and speed_day > 0.0:
                apply_sinking_ti(
                    self.tracers[name], 
                    dz.astype(np.float32), 
                    float(dt), 
                    float(speed_day), 
                    min_dz
                )