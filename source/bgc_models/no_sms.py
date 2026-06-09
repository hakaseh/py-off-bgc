import numpy as np
from dataclasses import dataclass
from numba import njit, prange
from .base import BaseBGCModel
from .utils import calculate_par

class Model_NO_SMS(BaseBGCModel):
    def __init__(self, nz, ny, nx, water_mask, params=None):
        super().__init__(nz, ny, nx, water_mask)
        
        # 1. Setup Tracers
        self.names = [
            'nitrate'
        ]
        self.tracers = {n: np.zeros((nz, ny, nx)) for n in self.names}
        
        # 2. Handle Parameters (The "Skip" Step)
        # If user gave nothing, create the default object. 
        # If user gave something, use it.
        if params is None:
            self.params = Params_BGC()
        else:
            self.params = params

    def biology_step(self, t_curr, sw_curr, dz, dt):
        """Step biology forward."""
        # 1. Calc Shared PAR
        #psum = 0 #self.tracers['PSn'] + self.tracers['PLn']
        par_3d = np.zeros_like(self.tracers['nitrate'])#calculate_par(sw_curr, psum, dz)

        return par_3d
    
    @staticmethod
    @njit(parallel=True, fastmath=True)
    def _run_kernel(
        nitrate, 
        temp, par, dz, dt,
        p_no_sms,
    ):
                
        nitrate_n = nitrate           
        
        return nitrate_n

@dataclass
class Params_BGC:
    """
    Parameter set for NEMURO.
    Default values from Table 1 Station A of Yoshie et al. 2007.
    The first two parameters (alpha1 and alpha2) are neglected.
    """
    p_no_sms: float = 0.0