import jax
import jax.numpy as jnp
from dataclasses import dataclass
from .base import BaseBGCModel
from .utils import calculate_par

@jax.jit
def fennel_kernel_jax(nitrate, 
                      ammonium, 
                      phy, 
                      chl, 
                      zoo, 
                      sdet, 
                      ldet, 
                      oxygen,
                      temp, 
                      par, 
                      dz, 
                      dt,
                      # Parameters
                      p_i_thnh4, 
                      p_d_p5nh4, 
                      p_nitrir, 
                      p_k_no3, 
                      p_k_nh4, 
                      p_vp0,
                      p_k_phy, 
                      p_phycn, 
                      p_phyis, 
                      p_phymr, 
                      p_chl2c_m, 
                      p_zooae_n,
                      p_zoobm, 
                      p_zooer, 
                      p_zoogr, 
                      p_zoomr, 
                      p_zoocn, 
                      p_lderrn,
                      p_sderrn, 
                      p_coagr, 
                      p_wp, 
                      p_ws, 
                      p_wl
                     ):
    
    dt_day = dt / 86400.0
    eps = 1e-12

    # --- D. SOURCES / SINKS ---
    
    # chl-to-carbon ratio
    cff = p_phycn * 12.0
    chl2c = jnp.minimum(chl / (phy * cff + eps), p_chl2c_m)

    # temperature- and light-limited growth rate (Eppley 1972)
    vp = p_vp0 * 0.59 * (1.066 ** temp)
    fac1_light = par * p_phyis
    epp = vp / jnp.sqrt(vp * vp + fac1_light * fac1_light)
    t_ppmax = epp * fac1_light

    # nutrient-limitation (Parker 1993)
    cff1_nut = ammonium * p_k_nh4
    cff2_nut = nitrate * p_k_no3
    inhnh4 = 1.0 / (1.0 + cff1_nut)
    l_nh4 = cff1_nut / (1.0 + cff1_nut)
    l_no3 = cff2_nut * inhnh4 / (1.0 + cff2_nut)
    ltot = l_no3 + l_nh4

    # nitrate and ammonium uptake by phytoplankton
    cff4_up = t_ppmax * p_k_no3 * inhnh4 / (1.0 + cff2_nut) * phy
    cff5_up = t_ppmax * p_k_nh4 / (1.0 + cff1_nut) * phy

    n_newprod = nitrate * cff4_up
    n_regprod = ammonium * cff5_up

    chl_prod = (t_ppmax * t_ppmax * ltot * ltot * p_chl2c_m * chl) / (p_phyis * jnp.maximum(chl2c, eps) * par + eps)

    # nitrification (nh4 --> no3, Olson 1981)
    fac2_o2 = jnp.maximum(oxygen, 0.0)
    fac3_o2 = jnp.maximum(fac2_o2 / (3.0 + fac2_o2), 0.0)
    fac1_nit = p_nitrir * fac3_o2

    cff1_nit = (par - p_i_thnh4) / (p_d_p5nh4 + par - 2.0 * p_i_thnh4)
    cff2_nit = 1.0 - jnp.maximum(cff1_nit, 0.0)
    cff3_nit = fac1_nit * cff2_nit
    n_nitrifi = ammonium * cff3_nit

    # if no light, no phyto growth and nitrification occurs at max rate
    no_light = par == 0.0
    n_newprod = jnp.where(no_light, 0.0, n_newprod)
    n_regprod = jnp.where(no_light, 0.0, n_regprod)
    chl_prod = jnp.where(no_light, 0.0, chl_prod)

    # zooplankton grazing (Landry 1993)
    cff1_graz = p_zoogr * zoo * phy / (p_k_phy + phy * phy)
    n_graz = cff1_graz * phy
    chl_graz = cff1_graz * chl

    n_assim = cff1_graz * phy * p_zooae_n
    n_egest = cff1_graz * phy * (1.0 - p_zooae_n)

    # phyto mortality
    n_pmortal = p_phymr * jnp.maximum(phy - 1e-6, 0.0)
    chl_pmortal = p_phymr * jnp.maximum(chl - 1e-6, 0.0)

    # zoo loss terms
    fac1_z = p_zooer * phy * phy / (p_k_phy + phy * phy)
    n_zmortal = p_zoomr * zoo * zoo
    n_zexcret = (fac1_z * p_zooae_n) * zoo
    n_zmetabo = p_zoobm * jnp.maximum(zoo - 1e-6, 0.0)

    # coagulation of phy and sdet to ldet
    cff1_coag = p_coagr * (sdet + phy)
    n_coagp = phy * cff1_coag
    chl_coag = chl * cff1_coag
    n_coagd = sdet * cff1_coag

    # remineralization
    fac1_rem = jnp.maximum(oxygen - 6.0, 0.0)
    fac2_rem = jnp.maximum(fac1_rem / (3.0 + fac1_rem), 0.0)
    n_remines = sdet * (p_sderrn * fac2_rem)
    n_reminel = ldet * (p_lderrn * fac2_rem)

    # --- E. DERIVATIVES ---
    d_nitrate = -n_newprod + n_nitrifi
    d_ammonium = -n_regprod - n_nitrifi + n_zmetabo + n_zexcret + n_remines + n_reminel
    d_phy = n_newprod + n_regprod - n_graz - n_pmortal - n_coagp
    d_chl = chl_prod - chl_graz - chl_pmortal - chl_coag
    d_zoo = n_assim - n_zmortal - n_zexcret - n_zmetabo
    d_sdet = n_egest + n_pmortal + n_zmortal - n_coagd - n_remines
    d_ldet = n_coagp + n_coagd - n_reminel
    d_oxygen = n_newprod * 8.625 + n_regprod * 6.625 - 2.0 * n_nitrifi - 6.625 * (n_zmetabo + n_zexcret + n_remines + n_reminel)

    # --- F. UPDATE ---
    # Apply geometric mask globally to filter out land/empty cells
    water_mask = dz > 1e-6

    return (
        jnp.where(water_mask, nitrate + d_nitrate * dt_day, nitrate),
        jnp.where(water_mask, ammonium + d_ammonium * dt_day, ammonium),
        jnp.where(water_mask, phy + d_phy * dt_day, phy),
        jnp.where(water_mask, chl + d_chl * dt_day, chl),
        jnp.where(water_mask, zoo + d_zoo * dt_day, zoo),
        jnp.where(water_mask, sdet + d_sdet * dt_day, sdet),
        jnp.where(water_mask, ldet + d_ldet * dt_day, ldet),
        jnp.where(water_mask, oxygen + d_oxygen * dt_day, oxygen)
    )

class Model_FENNEL06(BaseBGCModel):
    def __init__(self, nz, ny, nx, water_mask, params=None):
        super().__init__(nz, ny, nx, water_mask)
        
        # 1. Setup Tracers
        self.names = ['nitrate', 
                      'ammonium', 
                      'phy', 
                      'chl', 
                      'zoo', 
                      'sdet', 
                      'ldet', 
                      'oxygen'
                     ]
        
        # Swapped to jnp.zeros!
        self.tracers = {n: jnp.zeros((nz, ny, nx)) for n in self.names}
        
        # 2. Handle Parameters
        if params is None:
            self.params = Params_BGC()
        else:
            self.params = params

        # 3. Configure Sinking
        self.sinking_config = {'phy': self.params.p_wp,
                               'sdet': self.params.p_ws,
                               'ldet': self.params.p_wl
                              }

    def biology_step(self, t_curr, sw_curr, dz, dt):
        """Step biology forward."""
        # 1. Calc Shared PAR
        psum = self.tracers['chl']
        par_3d = calculate_par(sw_curr, psum, dz)

        # 2. Unpack Params
        p = self.params
        
        # 3. Call the JAX Kernel
        outputs = fennel_kernel_jax(
            # State variables
            self.tracers['nitrate'], 
            self.tracers['ammonium'], 
            self.tracers['phy'], 
            self.tracers['chl'], 
            self.tracers['zoo'],
            self.tracers['sdet'],
            self.tracers['ldet'],
            self.tracers['oxygen'],
            # Ancillary variables 
            t_curr, 
            par_3d, 
            dz, 
            dt,
            # Parameters (Explicitly passed)
            p.p_i_thnh4, 
            p.p_d_p5nh4, 
            p.p_nitrir, 
            p.p_k_no3, 
            p.p_k_nh4, 
            p.p_vp0,
            p.p_k_phy, 
            p.p_phycn, 
            p.p_phyis, 
            p.p_phymr, 
            p.p_chl2c_m, 
            p.p_zooae_n,
            p.p_zoobm, 
            p.p_zooer, 
            p.p_zoogr, 
            p.p_zoomr, 
            p.p_zoocn, 
            p.p_lderrn,
            p.p_sderrn, 
            p.p_coagr, 
            p.p_wp, 
            p.p_ws, 
            p.p_wl
         )
         
        # 4. Dictionary unpacking (Safe for JAX immutability)
        tracer_keys = ['nitrate', 'ammonium', 'phy', 'chl', 'zoo', 'sdet', 'ldet', 'oxygen']
        for key, jax_array in zip(tracer_keys, outputs):
            self.tracers[key] = jax_array
            
        return par_3d


@dataclass
class Params_BGC:
    """
    Parameter set for Fennel06.
    """
    p_i_thnh4: float = 0.0095
    p_d_p5nh4: float = 0.1
    p_nitrir: float = 0.2
    p_k_no3: float = 2.0
    p_k_nh4: float = 2.0
    p_vp0: float = 0.69
    p_k_phy: float = 0.5
    p_phycn: float = 6.625
    p_phyis: float = 0.025
    p_phymr: float = 0.072
    p_chl2c_m: float = 0.0535
    p_zooae_n: float = 0.75
    p_zoobm: float = 0.025
    p_zooer: float = 0.1
    p_zoogr: float = 0.75
    p_zoomr: float = 0.025
    p_zoocn: float = 6.625
    p_lderrn: float = 0.01
    p_sderrn: float = 0.03
    p_coagr: float = 0.005
    p_wp: float = 0.1
    p_ws: float = 0.1
    p_wl: float = 1.0