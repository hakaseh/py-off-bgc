import numpy as np
import taichi as ti
from dataclasses import dataclass
from .base import BaseBGCModel

# 1. Define the Taichi GPU Kernel
@ti.kernel
def biology_kernel_ti(
    nitrate: ti.types.ndarray(), 
    temp: ti.types.ndarray(), 
    par: ti.types.ndarray(), 
    dz: ti.types.ndarray(), 
    dt: ti.f32,
    p_no_sms: ti.f32
):
    nz, ny, nx = nitrate.shape
    
    # Massively parallel loop across the 3D grid
    for k, j, i in ti.ndrange(nz, ny, nx):
        # For an inert tracer, the Source-Minus-Sink (SMS) biological term is zero.
        # If this were active biology, you would calculate uptake/remineralization here:
        # sms_nitrate = ... 
        # nitrate[k, j, i] += sms_nitrate * dt
        pass


class Model_NO_SMS(BaseBGCModel):
    def __init__(self, nz, ny, nx, water_mask, params=None):
        super().__init__(nz, ny, nx, water_mask)
        
        # 1. Setup Tracers
        self.names = ['nitrate']
        self.tracers = {n: np.zeros((nz, ny, nx), dtype=np.float32) for n in self.names}
        
        # 2. Handle Parameters
        if params is None:
            self.params = Params_BGC()
        else:
            self.params = params

    def biology_step(self, t_curr, sw_curr, dz, dt):
        """Step biology forward."""
        
        # 1. Calc Shared PAR (Zeros since there is no phytoplankton shading)
        par_3d = np.zeros_like(self.tracers['nitrate'])
        
        # 2. Execute the Taichi biology kernel in-place
        biology_kernel_ti(
            self.tracers['nitrate'],
            t_curr,
            par_3d,
            dz,
            float(dt),
            float(self.params.p_no_sms)
        )

        return par_3d
        
    def sinking_step(self, dz, dt):
        """Step particulate sinking forward."""
        # Inert nitrate is dissolved, so it does not sink.
        pass


@dataclass
class Params_BGC:
    """
    Parameter set for the inert tracer model.
    """
    p_no_sms: float = 0.0