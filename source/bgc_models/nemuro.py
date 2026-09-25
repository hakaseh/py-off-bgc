import jax
import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass
from .base import BaseBGCModel
from .utils import calculate_par

@jax.jit
def nemuro_kernel_jax(
    PSn, PLn, ZSn, ZLn, ZPn, nitrate, ammonium, PON, DON, 
    PLsi, ZLsi, ZPsi, silicate, Opal, schl, lchl, oxygen,
    temp, par, dz, dt,
    # Parameters
    p_vmaxs, p_kno3s, p_knh4s, p_this, p_kgpps, p_iopts, p_resps0, 
    p_kresps, p_gs, p_morps0, p_kmorps, p_vmaxl, p_kno3l, p_knh4l, 
    p_ksil, p_thil, p_kgppl, p_ioptl, p_respl0, p_krespl, p_gl, 
    p_morpl0, p_kmorpl, p_grmaxsps, p_kgras, p_ls, p_ps2zs, 
    p_morzs0, p_kmorzs, p_alphazs, p_betazs, p_grmaxlps, p_grmaxlpl, 
    p_grmaxlzs, p_kgral, p_ll, p_ps2zl, p_pl2zl, p_zs2zl, p_morzl0, 
    p_kmorzl, p_alphazl, p_betazl, p_grmaxppl, p_grmaxpzs, p_grmaxpzl, 
    p_kgrap, p_lp, p_pl2zp, p_zs2zp, p_zl2zp, p_thipl, p_thizs, 
    p_morzp0, p_kmorzp, p_alphazp, p_betazp, p_nit0, p_knit, p_vp2n0, 
    p_kp2n, p_vp2d0, p_kp2d, p_vd2n0, p_kd2n, p_vp2si0, p_kp2si, 
    p_rsinpl, p_setvp, p_setvo, p_thetas, p_thetal, p_alphas, p_alphal
):
    dt_day = dt / 86400.0

    # --- D. SOURCES / SINKS (Entire 3D Ocean Simultaneously) ---

    # (1) Gross Primary Production rate of small phytoplankton
    GppPSn = p_vmaxs * (
        nitrate/(nitrate + p_kno3s) * jnp.exp(-p_this * ammonium)
        + ammonium/(ammonium + p_knh4s)
    ) * jnp.exp(p_kgpps * temp) * (par/p_iopts) * jnp.exp(1.0 - par/p_iopts) * PSn
    
    # f-ratio of small phytoplankton
    RnewS = (nitrate/(nitrate + p_kno3s) * jnp.exp(-p_this*ammonium)) / (
        nitrate/(nitrate + p_kno3s) * jnp.exp(-p_this*ammonium)
        + ammonium/(ammonium+p_knh4s) + 1e-12
    )
    
    # (2) Gross Primary Production rate of large phytoplankton                    
    N_lim = nitrate/(nitrate+p_kno3l)*jnp.exp(-p_thil*ammonium) + ammonium/(ammonium+p_knh4l)
    Si_lim = silicate/(silicate+p_ksil)/p_rsinpl
    
    GppPLn = p_vmaxl * jnp.minimum(N_lim, Si_lim) * jnp.exp(p_kgppl*temp) * (par/p_ioptl) * jnp.exp(1.0 - par/p_ioptl) * PLn
    
    # f-ratio of large phytoplankton
    RnewL = (nitrate/(nitrate + p_kno3l) * jnp.exp(-p_thil*ammonium)) / (
        nitrate/(nitrate + p_kno3l) * jnp.exp(-p_thil*ammonium)
        + ammonium/(ammonium+p_knh4l) + 1e-12
    )

    # (3-8) Respiration, Mortality, Excretion
    ResPSn = p_resps0 * jnp.exp(p_kresps * temp) * PSn
    ResPLn = p_respl0 * jnp.exp(p_krespl * temp) * PLn
    MorPSn = p_morps0 * jnp.exp(p_kmorps * temp) * PSn**2
    MorPLn = p_morpl0 * jnp.exp(p_kmorpl * temp) * PLn**2
    ExcPSn = p_gs * GppPSn
    ExcPLn = p_gl * GppPLn

    # (9-15) Grazing Rates
    GraPS2ZSn = jnp.maximum(0.0, p_grmaxsps * jnp.exp(p_kgras * temp) * (1.0 - jnp.exp(p_ls*(p_ps2zs - PSn))) * ZSn)
    GraPS2ZLn = jnp.maximum(0.0, p_grmaxlps * jnp.exp(p_kgral * temp) * (1.0 - jnp.exp(p_ll*(p_ps2zl - ZSn))) * ZLn)
    GraPL2ZLn = jnp.maximum(0.0, p_grmaxlpl * jnp.exp(p_kgral * temp) * (1.0 - jnp.exp(p_ll*(p_pl2zl - PLn))) * ZLn)
    GraZS2ZLn = jnp.maximum(0.0, p_grmaxlzs * jnp.exp(p_kgral * temp) * (1.0 - jnp.exp(p_ll*(p_zs2zl - ZSn))) * ZLn)
    
    GraPL2ZPn = jnp.maximum(0.0, p_grmaxppl * jnp.exp(p_kgrap * temp) * (1.0 - jnp.exp(p_lp*(p_pl2zp - PLn))) * ZPn * jnp.exp(-p_thipl*(ZLn + ZSn)))
    GraZS2ZPn = jnp.maximum(0.0, p_grmaxpzs * jnp.exp(p_kgrap * temp) * (1.0 - jnp.exp(p_lp*(p_zs2zp - ZSn))) * ZPn * jnp.exp(-p_thizs*ZLn))
    GraZL2ZPn = jnp.maximum(0.0, p_grmaxpzl * jnp.exp(p_kgrap * temp) * (1.0 - jnp.exp(p_lp*(p_zl2zp - ZLn))) * ZPn)

    # (16-24) Zooplankton Excretion, Egestion, Mortality
    ExcZSn = (p_alphazs - p_betazs) * GraPS2ZSn
    ExcZLn = (p_alphazl - p_betazl) * (GraPL2ZLn + GraZS2ZLn + GraPS2ZLn)
    ExcZPn = (p_alphazp - p_betazp) * (GraPL2ZPn + GraZS2ZPn + GraZL2ZPn) # Fixed duplicate GraPL2ZPn from original
    
    EgeZSn = (1.0 - p_alphazs) * GraPS2ZSn
    EgeZLn = (1.0 - p_alphazl) * (GraPL2ZLn + GraZS2ZLn + GraPS2ZLn)
    EgeZPn = (1.0 - p_alphazp) * (GraPL2ZPn + GraZS2ZPn + GraZL2ZPn)
    
    MorZSn = p_morzs0 * jnp.exp(p_kmorzs * temp) * ZSn**2
    MorZLn = p_morzl0 * jnp.exp(p_kmorzl * temp) * ZLn**2
    MorZPn = p_morzp0 * jnp.exp(p_kmorzp * temp) * ZPn**2

    # (25-28) Decomposition and Nitrification
    DecP2N = p_vp2n0 * jnp.exp(p_kp2n * temp) * PON 
    DecP2D = p_vp2d0 * jnp.exp(p_kp2d * temp) * PON 
    DecD2N = p_vd2n0 * jnp.exp(p_kd2n * temp) * DON 
    Nit    = p_nit0 * jnp.exp(p_knit * temp) * ammonium

    # (30-38) Silicon Cycle
    GppPLsi = GppPLn * p_rsinpl
    ResPLsi = ResPLn * p_rsinpl
    MorPLsi = MorPLn * p_rsinpl
    ExcPLsi = ExcPLn * p_rsinpl
    GraPL2ZLsi = GraPL2ZLn * p_rsinpl
    GraPL2ZPsi = GraPL2ZPn * p_rsinpl
    EgeZLsi = EgeZLn * p_rsinpl
    EgeZPsi = EgeZPn * p_rsinpl
    DecP2Si = p_vp2si0 * jnp.exp(p_kp2si * temp) * Opal

    # --- E. DERIVATIVES ---
    d_PSn = GppPSn - ResPSn - MorPSn - ExcPSn - GraPS2ZSn - GraPS2ZLn
    d_PLn = GppPLn - ResPLn - MorPLn - ExcPLn - GraPL2ZLn - GraPL2ZPn
    d_ZSn = GraPS2ZSn - GraZS2ZLn - GraZS2ZPn - MorZSn - ExcZSn - EgeZSn
    d_ZLn = GraPS2ZLn + GraPL2ZLn + GraZS2ZLn - GraZL2ZPn - MorZLn - ExcZLn - EgeZLn
    d_ZPn = GraPL2ZPn + GraZS2ZPn + GraZL2ZPn - MorZPn - ExcZPn - EgeZPn
    d_nitrate = - (GppPSn - ResPSn)*RnewS  - (GppPLn - ResPLn)*RnewL + Nit
    d_ammonium = - (GppPSn - ResPSn)*(1.0 - RnewS) - (GppPLn - ResPLn)*(1.0 - RnewL) - Nit + DecP2N + DecD2N + ExcZSn + ExcZLn + ExcZPn
    d_PON = MorPSn + MorPLn + MorZSn + MorZLn  + MorZPn + EgeZSn + EgeZLn + EgeZPn - DecP2N - DecP2D
    d_DON = ExcPSn + ExcPLn + DecP2D - DecD2N
    
    d_PLsi = GppPLsi - ResPLsi - MorPLsi - ExcPLsi - GraPL2ZLsi - GraPL2ZPsi
    d_ZLsi = GraPL2ZLsi - EgeZLsi
    d_ZPsi = GraPL2ZPsi - EgeZPsi
    d_silicate = - GppPLsi + ResPLsi + ExcPLsi + DecP2Si
    d_Opal = MorPLsi + EgeZLsi + EgeZPsi - DecP2Si

    # Chlorophyll-a
    # C:N mass conversion factor: (106 mol C / 16 mol N) * 12.01 mg C / mmol C ~= 79.57
    cn_mass = 106/16*12.01
    d_schl = p_thetas * GppPSn**2 / (PSn + 1e-12) / (p_alphas * par + 1e-12) * cn_mass - (ResPSn + MorPSn + ExcPSn + GraPS2ZSn + GraPS2ZLn) / (PSn + 1e-12) * schl     
    d_lchl = p_thetal * GppPLn**2 / (PLn + 1e-12) / (p_alphal * par + 1e-12) * cn_mass - (ResPLn + MorPLn + ExcPLn + GraPL2ZLn + GraPL2ZPn) / (PLn + 1e-12) * lchl      
    
    d_oxygen = - 172.0 / 16.0 * (d_nitrate + d_ammonium + Nit)

    # --- F. UPDATE & MASK ---
    # Apply updates to all cells unconditionally
    PSn_n = PSn + d_PSn * dt_day
    PLn_n = PLn + d_PLn * dt_day
    ZSn_n = ZSn + d_ZSn * dt_day
    ZLn_n = ZLn + d_ZLn * dt_day
    ZPn_n = ZPn + d_ZPn * dt_day
    nitrate_n = nitrate + d_nitrate * dt_day
    ammonium_n = ammonium + d_ammonium * dt_day
    PON_n = PON + d_PON * dt_day
    DON_n = DON + d_DON * dt_day
    PLsi_n = PLsi + d_PLsi * dt_day
    ZLsi_n = ZLsi + d_ZLsi * dt_day
    ZPsi_n = ZPsi + d_ZPsi * dt_day
    silicate_n = silicate + d_silicate * dt_day
    Opal_n = Opal + d_Opal * dt_day
    schl_n = schl + d_schl * dt_day
    lchl_n = lchl + d_lchl * dt_day                    
    oxygen_n = oxygen + d_oxygen * dt_day

    # Apply the land mask (dz > 1e-6) to ensure land pixels remain untouched
    water_mask = dz > 1e-6
    
    return (
        jnp.where(water_mask, PSn_n, PSn),
        jnp.where(water_mask, PLn_n, PLn),
        jnp.where(water_mask, ZSn_n, ZSn),
        jnp.where(water_mask, ZLn_n, ZLn),
        jnp.where(water_mask, ZPn_n, ZPn),
        jnp.where(water_mask, nitrate_n, nitrate),
        jnp.where(water_mask, ammonium_n, ammonium),
        jnp.where(water_mask, PON_n, PON),
        jnp.where(water_mask, DON_n, DON),
        jnp.where(water_mask, PLsi_n, PLsi),
        jnp.where(water_mask, ZLsi_n, ZLsi),
        jnp.where(water_mask, ZPsi_n, ZPsi),
        jnp.where(water_mask, silicate_n, silicate),
        jnp.where(water_mask, Opal_n, Opal),
        jnp.where(water_mask, schl_n, schl),
        jnp.where(water_mask, lchl_n, lchl),
        jnp.where(water_mask, oxygen_n, oxygen)
    )

class Model_NEMURO(BaseBGCModel):
    def __init__(self, nz, ny, nx, water_mask, params=None):
        # CORRECT: Do not pass 'params' into the super() call! 
        # BaseBGCModel only knows how to handle the grid dimensions.
        super().__init__(nz, ny, nx, water_mask)
        
        # 1. Setup Tracers
        self.names = [
            'PSn', 'PLn', 'ZSn', 'ZLn', 'ZPn', 'nitrate', 'ammonium', 'PON', 'DON', 
            'PLsi', 'ZLsi', 'ZPsi', 'silicate', 'Opal', 'schl', 'lchl', 'oxygen'
        ]
        self.tracers = {n: np.zeros((nz, ny, nx)) for n in self.names}
        
        # 2. Handle Parameters
        if params is None:
            self.params = Params_BGC()
        else:
            self.params = params

        # 3. Configure Sinking
        self.sinking_config = {
            'PON': self.params.p_setvp,
            'Opal': self.params.p_setvo
        }    
    def biology_step(self, t_curr, sw_curr, dz, dt):
        """Step biology forward using JAX."""
        # 1. Calc Shared PAR
        psum = self.tracers['schl'] + self.tracers['lchl']
        par_3d = calculate_par(sw_curr, psum, dz)

        # 2. Unpack Params
        p = self.params
        
        # 3. Call the external JAX Kernel
        outputs = nemuro_kernel_jax(
            self.tracers['PSn'], 
            self.tracers['PLn'], 
            self.tracers['ZSn'], 
            self.tracers['ZLn'], 
            self.tracers['ZPn'], 
            self.tracers['nitrate'], 
            self.tracers['ammonium'], 
            self.tracers['PON'], 
            self.tracers['DON'], 
            self.tracers['PLsi'],
            self.tracers['ZLsi'],
            self.tracers['ZPsi'],
            self.tracers['silicate'],
            self.tracers['Opal'],
            self.tracers['schl'],
            self.tracers['lchl'],
            self.tracers['oxygen'],
            t_curr, par_3d, dz, dt,
            # Parameters (Explicitly passed)
            p.p_vmaxs, p.p_kno3s, p.p_knh4s, p.p_this, p.p_kgpps, p.p_iopts, p.p_resps0, 
            p.p_kresps, p.p_gs, p.p_morps0, p.p_kmorps, p.p_vmaxl, p.p_kno3l, p.p_knh4l, 
            p.p_ksil, p.p_thil, p.p_kgppl, p.p_ioptl, p.p_respl0, p.p_krespl, p.p_gl, 
            p.p_morpl0, p.p_kmorpl, p.p_grmaxsps, p.p_kgras, p.p_ls, p.p_ps2zs, 
            p.p_morzs0, p.p_kmorzs, p.p_alphazs, p.p_betazs, p.p_grmaxlps, p.p_grmaxlpl, 
            p.p_grmaxlzs, p.p_kgral, p.p_ll, p.p_ps2zl, p.p_pl2zl, p.p_zs2zl, p.p_morzl0, 
            p.p_kmorzl, p.p_alphazl, p.p_betazl, p.p_grmaxppl, p.p_grmaxpzs, p.p_grmaxpzl, 
            p.p_kgrap, p.p_lp, p.p_pl2zp, p.p_zs2zp, p.p_zl2zp, p.p_thipl, p.p_thizs, 
            p.p_morzp0, p.p_kmorzp, p.p_alphazp, p.p_betazp, p.p_nit0, p.p_knit, p.p_vp2n0, 
            p.p_kp2n, p.p_vp2d0, p.p_kp2d, p.p_vd2n0, p.p_kd2n, p.p_vp2si0, p.p_kp2si, 
            p.p_rsinpl, p.p_setvp, p.p_setvo, p.p_thetas, p.p_thetal, p.p_alphas, p.p_alphal
        )
        
        # 4. Unpack JAX output tuple back into dictionary
        tracer_keys = [
            'PSn', 'PLn', 'ZSn', 'ZLn', 'ZPn', 'nitrate', 'ammonium', 'PON', 'DON', 
            'PLsi', 'ZLsi', 'ZPsi', 'silicate', 'Opal', 'schl', 'lchl', 'oxygen'
        ]
        for key, jax_array in zip(tracer_keys, outputs):
            self.tracers[key] = jax_array
            
        return par_3d

@dataclass
class Params_BGC:
    """
    Parameter set for NEMURO.
    Default values from Table 1 Station A of Yoshie et al. 2007.
    The first two parameters (alpha1 and alpha2) are neglected.
    """
    # Parameters for phy1
    p_vmaxs: float = 0.4 # 
    p_kno3s: float = 1.0 # 
    p_knh4s: float = 0.1 # 
    p_this: float = 1.5 # 
    p_kgpps: float = 0.0693 # 
    p_iopts: float = 104.7 # 
    p_resps0: float = 0.03 # 
    p_kresps: float = 0.0519
    p_gs: float = 0.135
    p_morps0: float = 0.0585
    p_kmorps: float = 0.0693
    # Parameters for phy2
    p_vmaxl: float = 0.8 # 
    p_kno3l: float = 3.0 # 
    p_knh4l: float = 0.3 # 
    p_ksil: float = 6.0 # 
    p_thil: float = 1.5 # 
    p_kgppl: float = 0.0693 # 
    p_ioptl: float = 104.7 # 
    p_respl0: float = 0.03
    p_krespl: float = 0.0519
    p_gl: float = 0.135
    p_morpl0: float = 0.029
    p_kmorpl: float = 0.0693
    # Parameters for small zooplankton (ZS)
    p_grmaxsps: float = 0.4
    p_kgras: float = 0.0693
    p_ls: float = 1.4
    p_ps2zs: float = 0.04
    p_morzs0: float = 0.0585
    p_kmorzs: float = 0.0693
    p_alphazs: float = 0.7
    p_betazs: float = 0.3
    # Parameters for large zooplankton (ZL)
    p_grmaxlps: float = 0.1
    p_grmaxlpl: float = 0.4
    p_grmaxlzs: float = 0.4
    p_kgral: float = 0.0693
    p_ll: float = 1.4
    p_ps2zl: float = 0.04
    p_pl2zl: float = 0.04
    p_zs2zl: float = 0.04
    p_morzl0: float = 0.0585
    p_kmorzl: float = 0.0693
    p_alphazl: float = 0.7
    p_betazl: float = 0.3
    # Parameters for predatory zooplankton (ZP)
    p_grmaxppl: float = 0.2
    p_grmaxpzs: float = 0.2
    p_grmaxpzl: float = 0.2
    p_kgrap: float = 0.0693
    p_lp: float = 1.4
    p_pl2zp: float = 0.04
    p_zs2zp: float = 0.04
    p_zl2zp: float = 0.04
    p_thipl: float = 4.605
    p_thizs: float = 3.01
    p_morzp0: float = 0.0585
    p_kmorzp: float = 0.0693
    p_alphazp: float = 0.7
    p_betazp: float = 0.3
    # Other parameters (decomposition, etc.)
    p_nit0: float = 0.03
    p_knit: float = 0.0693
    p_vp2n0: float = 0.1
    p_kp2n: float = 0.0693
    p_vp2d0: float = 0.1
    p_kp2d: float = 0.0693
    p_vd2n0: float = 0.02
    p_kd2n: float = 0.0693
    p_vp2si0: float = 0.1
    p_kp2si: float = 0.0693
    p_rsinpl: float = 2.0 # Si:N ratio of plankton
    p_setvp: float = 40
    p_setvo: float = 40
    p_thetas: float = 0.0328 # Table S1, Laurent et al. 2021
    p_thetal: float = 0.0386 # Table S1, Laurent et al. 2021
    p_alphas: float = 0.0405 # Table S1, Laurent et al. 2021
    p_alphal: float = 0.0393 # Table S1, Laurent et al. 2021