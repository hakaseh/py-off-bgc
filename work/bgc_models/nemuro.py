import numpy as np
from dataclasses import dataclass
from numba import njit, prange
from .base import BaseBGCModel
from .utils import calculate_par

class Model_NEMURO(BaseBGCModel):
    def __init__(self, nz, ny, nx, water_mask, params=None):
        super().__init__(nz, ny, nx, water_mask)
        
        # 1. Setup Tracers
        self.names = ['nitrate', 'ammonium', 'silicate', 
                      'phy1', 'phy2', 'zoo1', 'zoo2', 'zoo3',
                      'pon', 'don', 'opal']
        self.tracers = {n: np.zeros((nz, ny, nx)) for n in self.names}
        
        # 2. Handle Parameters (The "Skip" Step)
        # If user gave nothing, create the default object. 
        # If user gave something, use it.
        if params is None:
            self.params = Params_NEMURO()
        else:
            self.params = params

        # 3. Configure Sinking
        # Read directly from self.params. It is guaranteed to work now.
        self.sinking_config = {
            'pon': self.params.p1_detr,
            'opal': self.params.p4_detr
        }

    def biology_step(self, t_curr, sw_curr, dz, dt):
        """Step biology forward."""
        # 1. Calc Shared PAR
        psum = self.tracers['phy1'] + self.tracers['phy2']
        par_3d = calculate_par(sw_curr, psum, dz)

        # 2. Unpack Params for Numba 
        # (Numba cannot read self.params, we must pass floats)
        p = self.params
        
        # 2. Call Specific Kernel
        (
         self.tracers['nitrate'], 
         self.tracers['ammonium'], 
         self.tracers['silicate'], 
         self.tracers['phy1'], 
         self.tracers['phy2'], 
         self.tracers['zoo1'], 
         self.tracers['zoo2'], 
         self.tracers['zoo3'], 
         self.tracers['pon'], 
         self.tracers['don'], 
         self.tracers['opal']
        ) = self._run_kernel(
             self.tracers['nitrate'], 
             self.tracers['ammonium'], 
             self.tracers['silicate'], 
             self.tracers['phy1'], 
             self.tracers['phy2'], 
             self.tracers['zoo1'], 
             self.tracers['zoo2'], 
             self.tracers['zoo3'], 
             self.tracers['pon'], 
             self.tracers['don'], 
             self.tracers['opal'],
             t_curr, par_3d, dz, dt,
             # Parameters (Explicitly passed)
             p.p_res, p.p_pmo, p.p_exc, p.p_gra, p.p_ivl, p.p_p2z,
             p.p_aef, p.p_gef, p.p_zmo, p.p_dec, p.p_gro, p.p_upt,
             p.p_aff, p.p_ini, p.p_the, p.p_act, p.p_gas, p.p_ref, p.p_sin
         )
        return par_3d
    
    @staticmethod
    @njit(parallel=True, fastmath=True)
    def _run_kernel(nitrate, ammonium, silicate, phy1, phy2, zoo1, zoo2, zoo3,
                    pon, don, opal, temp, par, dz, dt,
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
        ammonium_new = np.zeros_like(ammonium)
        silicate_new = np.zeros_like(silicate)
        phy1_new  = np.zeros_like(phy1)
        phy2_new  = np.zeros_like(phy2)
        zoo1_new  = np.zeros_like(zoo1)
        zoo2_new  = np.zeros_like(zoo2)
        zoo3_new  = np.zeros_like(zoo3)
        pon_new   = np.zeros_like(pon)
        don_new   = np.zeros_like(don)
        opal_new  = np.zeros_like(opal)
    
        for j in prange(ny):
            for i in range(nx):
                if dz[0, j, i] <= 1e-6: continue
    
                for k in range(nz):
                    if dz[k, j, i] <= 1e-6: continue
                    
                    # Load State
                    nitrate_cell = nitrate[k, j, i]
                    ammonium_cell = ammonium[k, j, i]
                    silicate_cell = silicate[k, j, i]
                    phy1_cell  = phy1[k, j, i]
                    phy2_cell  = phy2[k, j, i]
                    zoo1_cell  = zoo1[k, j, i]
                    zoo2_cell  = zoo2[k, j, i]
                    zoo3_cell  = zoo3[k, j, i]
                    pon_cell  = pon[k, j, i]
                    don_cell  = don[k, j, i]
                    opal_cell  = opal[k, j, i]
                    temp_cell   = temp[k, j, i]
                    par_cell = par[k, j, i]                
                    
                    # --- D. SOURCES / SINKS ---
                    gpp_ps_n = vmaxs * ( 
                        nitrate_cell/(nitrate_cell+kno3)
                        * exp(-thi_s*ammonium_cell)
                        + ammonium_cell/(ammonium_cell+knh4)
                        ) * exp(kgpps*temp_cell)*par_cell/iopts*exp(1-par_cell/iopts)*
                    # Realized Growth Rate
                    lim_tem = np.exp(-p_act/p_gas*(1.0/(temp_cell+273.15)-1.0/(p_ref+273.15)))
                    lim_lig = 1.0 - np.exp(-p_ini*p_the*par_cell/p_gro/lim_tem)
                    r_gro = p_gro * nitrate_cell / (
                        nitrate_cell + p_upt/p_aff + 
                        2*np.sqrt(p_upt*nitrate_cell/p_aff)
                    ) * lim_lig * lim_tem * phy_cell
                    r_pre = p_res * phy_cell
                    r_pmo = p_pmo * phy_cell * phy_cell
                    r_pex = p_exc * r_gro
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
    Parameter set for NEMURO.
    Default values from Table 1 Station A of Yoshie et al. 2007.
    The first two parameters (alpha1 and alpha2) are neglected.
    """
    # Parameters for phy1
    p1_phy1: float = 0.4 # 
    p2_phy1: float = 1.0 # 
    p3_phy1: float = 0.1 # 
    p4_phy1: float = 1.5 # 
    p5_phy1: float = 0.0693 # 
    p6_phy1: float = 104.7 # 
    p7_phy1: float = 0.0585 # 
    p8_phy1: float = 0.0693
    p9_phy1: float = 0.03
    p10_phy1: float = 0.0519
    p11_phy1: float = 0.135
    # Parameters for phy2
    p1_phy2: float = 0.8 # 
    p2_phy2: float = 3.0 # 
    p3_phy2: float = 0.3 # 
    p4_phy2: float = 6.0 # 
    p5_phy2: float = 1.5 # 
    p6_phy2: float = 0.0693 # 
    p7_phy2: float = 104.7 # 
    p8_phy2: float = 0.029
    p9_phy2: float = 0.0693
    p10_phy2: float = 0.03
    p11_phy2: float = 0.0519
    p12_phy2: float = 0.135
    # Parameters for zoo1
    p1_zoo1: float = 0.4
    p2_zoo1: float = 0.0693
    p3_zoo1: float = 1.4
    p4_zoo1: float = 0.043
    p5_zoo1: float = 0.7
    p6_zoo1: float = 0.3
    p7_zoo1: float = 0.0585
    p8_zoo1: float = 0.0693
    # Parameters for zoo2
    p1_zoo2: float = 0.1
    p2_zoo2: float = 0.4
    p3_zoo2: float = 0.4
    p4_zoo2: float = 0.0693
    p5_zoo2: float = 1.4
    p6_zoo2: float = 0.04
    p7_zoo2: float = 0.04
    p8_zoo2: float = 0.04
    p9_zoo2: float = 0.7
    p10_zoo2: float = 0.3
    p11_zoo2: float = 0.0585
    p12_zoo2: float = 0.0693
    # Parameters for zoo3
    p1_zoo3: float = 0.2
    p2_zoo3: float = 0.2
    p3_zoo3: float = 0.2
    p4_zoo3: float = 0.0693
    p5_zoo3: float = 1.4
    p6_zoo3: float = 0.04
    p7_zoo3: float = 0.04
    p8_zoo3: float = 0.04
    p9_zoo3: float = 4.605
    p10_zoo3: float = 3.01
    p11_zoo3: float = 0.7
    p12_zoo3: float = 0.3
    p13_zoo3: float = 0.0585
    p14_zoo3: float = 0.0693
    # Parameters for nitrification
    p1_nitr: float = 0.03
    p2_nitr: float = 0.0693
    # Parameters for sinking and decomposition
    p1_detr: float = 40
    p2_detr: float = 0.1
    p3_detr: float = 0.0693
    p4_detr: float = 0.1
    p5_detr: float = 0.0693
    p6_detr: float = 0.2
    p7_detr: float = 0.0693
    p8_detr: float = 40
    p9_detr: float = 0.1
    p10_detr: float = 0.0693
    # Parameter for ratio
    p1_s2n: float 2.0