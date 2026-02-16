import numpy as np
from dataclasses import dataclass
from numba import njit, prange
from .base import BaseBGCModel
from .utils import calculate_par

class Model_NPZD(BaseBGCModel):
    def __init__(self, nz, ny, nx, water_mask, params=None):
        super().__init__(nz, ny, nx, water_mask)
        
        # 1. Setup Tracers
        self.names = ['nitrate', 'phy', 'zoo', 'det', 'oxygen']
        self.tracers = {n: np.zeros((nz, ny, nx)) for n in self.names}
        
        # 2. Handle Parameters (The "Skip" Step)
        # If user gave nothing, create the default object. 
        # If user gave something, use it.
        if params is None:
            self.params = Params_NPZD()
        else:
            self.params = params

        # 3. Configure Sinking
        # Read directly from self.params. It is guaranteed to work now.
        self.sinking_config = {
            'det': self.params.p_sin
        }

    def biology_step(self, t_curr, sw_curr, dz, dt):
        """Step biology forward."""
        # 1. Calc Shared PAR
        psum = self.tracers['phy']
        par_3d = calculate_par(sw_curr, psum, dz)

        # 2. Unpack Params for Numba 
        # (Numba cannot read self.params, we must pass floats)
        p = self.params
        
        # 2. Call Specific Kernel
        (self.tracers['nitrate'], 
         self.tracers['phy'], 
         self.tracers['zoo'], 
         self.tracers['det'], 
         self.tracers['oxygen']) = self._run_kernel(
             self.tracers['nitrate'], 
             self.tracers['phy'], 
             self.tracers['zoo'], 
             self.tracers['det'],
             self.tracers['oxygen'],
             t_curr, par_3d, dz, dt,
             # Parameters (Explicitly passed)
             p.p_res, p.p_pmo, p.p_exc, p.p_gra, p.p_ivl, p.p_p2z,
             p.p_aef, p.p_gef, p.p_zmo, p.p_dec, p.p_gro, p.p_upt,
             p.p_aff, p.p_ini, p.p_the, p.p_act, p.p_gas, p.p_ref, p.p_sin
         )
        return par_3d
    
    @staticmethod
    @njit(parallel=True, fastmath=True)
    def _run_kernel(nitrate, phy, zoo, det, oxygen, temp, par, dz, dt,
                    # Parameters as arguments
                    p_res, p_pmo, p_exc, p_gra, p_ivl, p_p2z,
                    p_aef, p_gef, p_zmo, p_dec, p_gro, p_upt,
                    p_aff, p_ini, p_the, p_act, p_gas, p_ref, p_sin):

        nz, ny, nx = nitrate.shape
        
        # Time conversion (dt expressed in day, which is easy to deal with because
        # BGC parameters are often expressed in per day)
        dt_day = dt / 86400.0
        
        # Initialize Outputs
        nitrate_new = np.zeros_like(nitrate)
        phy_new = np.zeros_like(phy)
        zoo_new  = np.zeros_like(zoo)
        det_new  = np.zeros_like(det)
        oxygen_new  = np.zeros_like(oxygen)
    
        for j in prange(ny):
            for i in range(nx):
                if dz[0, j, i] <= 1e-6: continue
    
                for k in range(nz):
                    if dz[k, j, i] <= 1e-6: continue
                    
                    # Load State
                    nitrate_cell = nitrate[k, j, i]
                    phy_cell  = phy[k, j, i]
                    zoo_cell  = zoo[k, j, i]
                    det_cell  = det[k, j, i]
                    oxygen_cell  = oxygen[k, j, i]
                    temp_cell   = temp[k, j, i]
                    par_cell = par[k, j, i]                
                    
                    # --- D. SOURCES / SINKS ---
                    # Realized Growth Rate
                    lim_tem = np.exp(-p_act/p_gas*(1.0/(temp_cell+298.0)-1.0/(p_ref+298.0)))
                    lim_lig = 1.0 - np.exp(-p_ini*p_the*par_cell/p_gro/lim_tem)
                    r_gro = p_gro * nitrate_cell / (
                        nitrate_cell + p_upt/p_aff + 
                        2*np.sqrt(p_upt*nitrate_cell/p_aff)
                    ) * lim_lig * lim_tem * phy_cell
                    r_pre = p_res * phy_cell
                    r_pmo = p_pmo * phy_cell * phy_cell
                    r_pex = p_exc * r_gro
                    # check with sasai-san but i think Z in A6 is a typo?
                    r_gra = p_gra * (1.0 - np.exp(p_ivl*(p_p2z-phy_cell))) * zoo_cell
                    r_zmo = p_zmo * zoo_cell * zoo_cell
                    r_dec = p_dec * det_cell
    
                    
                    # --- E. DERIVATIVES ---
                    
                    d_nitrate = - r_gro + r_pre + r_pex + (p_aef - p_gef)*r_gra + r_dec
                    d_phy = r_gro - r_pre - r_pmo - r_pex - r_gra
                    d_zoo = r_gra - (p_aef - p_gef)*r_gra - (1 - p_aef)*r_gra - r_zmo
                    # sinking is done in anotehr routine
                    d_det = r_pmo + (1.0 - p_aef)*r_gra + r_zmo - r_dec
                    d_oxygen = - 172.0 / 16.0 * d_nitrate
                    
                    # --- F. UPDATE ---
                    nitrate_new[k, j, i] = nitrate_cell + d_nitrate * dt_day
                    phy_new[k, j, i] = phy_cell + d_phy * dt_day
                    zoo_new[k, j, i] = zoo_cell + d_zoo * dt_day
                    det_new[k, j, i] = det_cell + d_det * dt_day
                    oxygen_new[k, j, i] = oxygen_cell + d_oxygen * dt_day
        
        return nitrate_new, phy_new, zoo_new, det_new, oxygen_new


@dataclass
class Params_NPZD:
    """
    Configuration object for NPZD Model.
    Default values from Sasai et al 2022.
    """
    p_res: float = 0.12 # respiration rate (1/d)
    p_pmo: float = 0.24 # phy mortality rate (1/d)
    p_exc: float = 0.135 # extracellular excretion rate
    p_gra: float = 1.4 # maximum grazing rate (1/d)
    p_ivl: float = 1.4 # ivlev constant (mmol/m3)
    p_p2z: float = 0.04
    p_aef: float = 0.7 # assimilation efficiency (-)
    p_gef: float = 0.3 # growth efficiency (-)
    p_zmo: float = 0.12 # zooplankton mortality rate (1/d)
    p_dec: float = 0.3 # decomposition rate (1/d)
    p_gro: float = 1.5 # maximum growth rate (1/d)
    p_upt: float = 1 # maximum uptake rate (1/d)
    p_aff: float = 2 # maximum affinity
    p_ini: float = 2 # chl-specific initial slope of growth (-)
    p_the: float = 0.6 # chl-to-carbon ratio
    p_act: float = 4.8e4 # activation energy
    p_gas: float = 8.3145 # gas constant
    p_ref: float = 20 # reference temperature (degC)
    p_sin: float = 30 # detritus sinking speed (m/d)

