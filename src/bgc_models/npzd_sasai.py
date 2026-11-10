import numpy as np
from dataclasses import dataclass
from .base import BaseBGCModel
from .utils import calculate_par
import jax
import jax.numpy as jnp

@jax.jit
def npzd_kernel_jax(
    nitrate, phy, zoo, det, oxygen, temp, par, dz, dt,
    # Parameters
    p_res, p_pmo, p_exc, p_gra, p_ivl, p_p2z,
    p_aef, p_gef, p_zmo, p_dec, p_gro, p_upt,
    p_aff, p_ini, p_the, p_act, p_gas, p_ref, p_sin
):
    # Time conversion (dt expressed in day)
    dt_day = dt / 86400.0
    
    # --- D. SOURCES / SINKS (Calculated for the entire 3D array instantly) ---
    
    # Realized Growth Rate
    lim_tem = jnp.exp(-p_act/p_gas * (1.0/(temp + 273.15) - 1.0/(p_ref + 273.15)))
    lim_lig = 1.0 - jnp.exp(-p_ini * p_the * par / p_gro / lim_tem)
    
    # Using jnp.sqrt for the array-wise square root
    r_gro = p_gro * nitrate / (
        nitrate + p_upt/p_aff + 
        2.0 * jnp.sqrt(p_upt * nitrate / p_aff)
    ) * lim_lig * lim_tem * phy
    
    r_pre = p_res * phy
    r_pmo = p_pmo * phy * phy
    r_pex = p_exc * r_gro
    r_gra = p_gra * (1.0 - jnp.exp(p_ivl * (p_p2z - phy))) * zoo
    r_zmo = p_zmo * zoo * zoo
    r_dec = p_dec * det

    # --- E. DERIVATIVES ---
    d_nitrate = - r_gro + r_pre + r_pex + (p_aef - p_gef)*r_gra + r_dec
    d_phy = r_gro - r_pre - r_pmo - r_pex - r_gra
    d_zoo = r_gra - (p_aef - p_gef)*r_gra - (1.0 - p_aef)*r_gra - r_zmo
    d_det = r_pmo + (1.0 - p_aef)*r_gra + r_zmo - r_dec
    d_oxygen = - 172.0 / 16.0 * d_nitrate
    
    # --- F. UPDATE ---
    # Apply updates unconditionally to all cells
    nitrate_new = nitrate + d_nitrate * dt_day
    phy_new = phy + d_phy * dt_day
    zoo_new = zoo + d_zoo * dt_day
    det_new = det + d_det * dt_day
    oxygen_new = oxygen + d_oxygen * dt_day
    
    # --- G. MASKING ---
    # Only keep the updated values where it is actual ocean water (dz > 1e-6)
    water_mask = dz > 1e-6
    
    return (
        jnp.where(water_mask, nitrate_new, nitrate),
        jnp.where(water_mask, phy_new, phy),
        jnp.where(water_mask, zoo_new, zoo),
        jnp.where(water_mask, det_new, det),
        jnp.where(water_mask, oxygen_new, oxygen)
    )

class Model_NPZD(BaseBGCModel):
    def __init__(self, nz, ny, nx, water_mask, params=None):
        super().__init__(nz, ny, nx, water_mask)
        
        # 1. Setup Tracers
        self.names = ['nitrate', 'phy', 'zoo', 'det', 'oxygen']
        self.tracers = {n: np.zeros((nz, ny, nx)) for n in self.names}
        
        # 2. Handle Parameters
        if params is None:
            self.params = Params_BGC()
        else:
            self.params = params

        # 3. Configure Sinking
        self.sinking_config = {
            'det': self.params.p_sin
        }

    def biology_step(self, t_curr, sw_curr, dz, dt):
        """Step biology forward using JAX."""
        # 1. Calc Shared PAR
        psum = self.tracers['phy']
        par_3d = calculate_par(sw_curr, psum, dz)

        # 2. Unpack Params
        p = self.params
        
        # 3. Call Specific Kernel
        outputs = npzd_kernel_jax(
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
         
        # 4. Pack the returned tuple back into self.tracers (JAX Immutability Safe)
        tracer_keys = ['nitrate', 'phy', 'zoo', 'det', 'oxygen']
        for key, jax_array in zip(tracer_keys, outputs):
            self.tracers[key] = jax_array
            
        return par_3d


@dataclass
class Params_BGC:
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