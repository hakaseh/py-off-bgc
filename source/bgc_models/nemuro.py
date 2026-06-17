import numpy as np
from dataclasses import dataclass
from numba import njit, prange
from .base import BaseBGCModel
from .utils import calculate_par

class Model_NEMURO(BaseBGCModel):
    def __init__(self, nz, ny, nx, water_mask, params=None):
        super().__init__(nz, ny, nx, water_mask)
        
        # 1. Setup Tracers
        self.names = [
            'PSn', 
            'PLn', 
            'ZSn', 
            'ZLn', 
            'ZPn',
            'nitrate', 
            'ammonium', 
            'PON', 
            'DON', 
            'PLsi', 
            'ZLsi', 
            'ZPsi', 
            'silicate',
            'Opal',
            'schl',
            'lchl',
            'oxygen'
        ]
        self.tracers = {n: np.zeros((nz, ny, nx)) for n in self.names}
        
        # 2. Handle Parameters (The "Skip" Step)
        # If user gave nothing, create the default object. 
        # If user gave something, use it.
        if params is None:
            self.params = Params_BGC()
        else:
            self.params = params

        # 3. Configure Sinking
        # Read directly from self.params.
        self.sinking_config = {
            'PON': self.params.p_setvp,
            'Opal': self.params.p_setvo
        }

    def biology_step(self, t_curr, sw_curr, dz, dt):
        """Step biology forward."""
        # 1. Calc Shared PAR
        psum = self.tracers['schl'] + self.tracers['lchl']
        par_3d = calculate_par(sw_curr, psum, dz)

        # 2. Unpack Params for Numba 
        # (Numba cannot read self.params, we must pass floats)
        p = self.params
        
        # 2. Call Specific Kernel
        ( 
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
         self.tracers['oxygen']
        ) = self._run_kernel(
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
            p.p_vmaxs,
            p.p_kno3s,
            p.p_knh4s,
            p.p_this,
            p.p_kgpps,
            p.p_iopts,
            p.p_resps0,
            p.p_kresps,
            p.p_gs,
            p.p_morps0,
            p.p_kmorps,
            p.p_vmaxl,
            p.p_kno3l,
            p.p_knh4l,
            p.p_ksil,
            p.p_thil,
            p.p_kgppl,
            p.p_ioptl,
            p.p_respl0,
            p.p_krespl,
            p.p_gl,
            p.p_morpl0,
            p.p_kmorpl,
            p.p_grmaxsps,
            p.p_kgras,
            p.p_ls,
            p.p_ps2zs,
            p.p_morzs0,
            p.p_kmorzs,
            p.p_alphazs,
            p.p_betazs,
            p.p_grmaxlps,
            p.p_grmaxlpl,
            p.p_grmaxlzs,
            p.p_kgral,
            p.p_ll,
            p.p_ps2zl,
            p.p_pl2zl,
            p.p_zs2zl,
            p.p_morzl0,
            p.p_kmorzl,
            p.p_alphazl,
            p.p_betazl,
            p.p_grmaxppl,
            p.p_grmaxpzs,
            p.p_grmaxpzl,
            p.p_kgrap,
            p.p_lp,
            p.p_pl2zp,
            p.p_zs2zp,
            p.p_zl2zp,
            p.p_thipl,
            p.p_thizs,
            p.p_morzp0,
            p.p_kmorzp,
            p.p_alphazp,
            p.p_betazp,
            p.p_nit0,
            p.p_knit,
            p.p_vp2n0,
            p.p_kp2n,
            p.p_vp2d0,
            p.p_kp2d,
            p.p_vd2n0,
            p.p_kd2n,
            p.p_vp2si0,
            p.p_kp2si,
            p.p_rsinpl,
            p.p_setvp,
            p.p_setvo,
            p.p_thetas,
            p.p_thetal,
            p.p_alphas,
            p.p_alphal
         )
        return par_3d
    
    @staticmethod
    @njit(parallel=True, fastmath=True)
    def _run_kernel(
        PSn, PLn, ZSn, ZLn, ZPn, nitrate, ammonium, PON, DON, 
        PLsi, ZLsi, ZPsi, silicate, Opal, schl, lchl, oxygen,
        temp, par, dz, dt,
        # Parameters as arguments
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

        nz, ny, nx = nitrate.shape
        
        # Time conversion (dt expressed in day)
        dt_day = dt / 86400.0
        
        # Initialize Outputs using .copy() to preserve land values
        PSn_n = PSn.copy()
        PLn_n = PLn.copy()
        ZSn_n = ZSn.copy()
        ZLn_n = ZLn.copy()
        ZPn_n = ZPn.copy()
        nitrate_n = nitrate.copy()
        ammonium_n = ammonium.copy()
        PON_n = PON.copy()
        DON_n = DON.copy()
        PLsi_n = PLsi.copy()
        ZLsi_n = ZLsi.copy()
        ZPsi_n = ZPsi.copy()
        silicate_n = silicate.copy()
        Opal_n = Opal.copy()
        schl_n = schl.copy()
        lchl_n = lchl.copy()
        oxygen_n = oxygen.copy()

        # 1. Calculate total number of horizontal grid points
        n_points = ny * nx
        
        # 2. Single flattened loop for OpenMP thread distribution
        for p in prange(n_points):
            # 3. Reconstruct 2D spatial indices (j, i)
            j = p // nx
            i = p % nx
            
            if dz[0, j, i] <= 1e-6: 
                continue
    
            for k in range(nz):
                if dz[k, j, i] <= 1e-6: 
                    continue
                    
                # Load State
                PSn_c = PSn[k, j, i]
                PLn_c = PLn[k, j, i]
                ZSn_c = ZSn[k, j, i]
                ZLn_c = ZLn[k, j, i]
                ZPn_c = ZPn[k, j, i]
                nitrate_c = nitrate[k, j, i]
                ammonium_c = ammonium[k, j, i]
                PON_c = PON[k, j, i]
                DON_c = DON[k, j, i]
                PLsi_c = PLsi[k, j, i]
                ZLsi_c = ZLsi[k, j, i]
                ZPsi_c = ZPsi[k, j, i]
                silicate_c = silicate[k, j, i]
                Opal_c = Opal[k, j, i]  
                schl_c = schl[k, j, i]  
                lchl_c = lchl[k, j, i]  
                oxygen_c = oxygen[k, j, i]                

                temp_c = temp[k, j, i]
                par_c = par[k, j, i]                
                
                # --- D. SOURCES / SINKS ---

                # (1) Gross Primary Production rate of small phytoplankton
                GppPSn = p_vmaxs * (
                    nitrate_c/(nitrate_c + p_kno3s)
                    * np.exp(-p_this * ammonium_c)
                    + ammonium_c/(ammonium_c + p_knh4s)
                    ) * np.exp(p_kgpps * temp_c
                    ) * par_c/p_iopts * np.exp(1.0 - par_c/p_iopts) * PSn_c
                # f-ratio of small phytoplankton
                RnewS = nitrate_c/(nitrate_c + p_kno3s)*np.exp(-p_this*ammonium_c
                    )/(nitrate_c/(nitrate_c + p_kno3s)*np.exp(-p_this*ammonium_c)
                    +ammonium_c/(ammonium_c+p_knh4s) + 1e-12)
                # (2) Gross Primary Production rate of large phytoplankton                    
                GppPLn = p_vmaxl * min(
                    nitrate_c/(nitrate_c+p_kno3l)
                    *np.exp(-p_thil*ammonium_c)
                    +ammonium_c/(ammonium_c+p_knh4l),
                    silicate_c/(silicate_c+p_ksil)/p_rsinpl
                    )*np.exp(p_kgppl*temp_c)*par_c/p_ioptl*np.exp(
                    1.0 - par_c/p_ioptl) * PLn_c
                # f-ratio of large phytoplankton
                RnewL = nitrate_c/(nitrate_c + p_kno3l)*np.exp(-p_thil*ammonium_c
                    )/(nitrate_c/(nitrate_c + p_kno3l)*np.exp(-p_thil*ammonium_c)
                    +ammonium_c/(ammonium_c+p_knh4l) + 1e-12)
                # (3) Respiration rate of small phytoplankton
                ResPSn = p_resps0*np.exp(p_kresps*temp_c) * PSn_c
                # (4) Respiration rate of large phytoplankton
                ResPLn = p_respl0*np.exp(p_krespl*temp_c) * PLn_c
                # (5) Mortality rate of small phytoplankton
                MorPSn = p_morps0*np.exp(p_kmorps*temp_c) * PSn_c**2
                # (6) Mortality rate of large phytoplankton
                MorPLn = p_morpl0*np.exp(p_kmorpl*temp_c) * PLn_c**2
                # (7) Extracellular excretion rate of small phytoplankton
                ExcPSn = p_gs*GppPSn
                # (8) Extracellular excretion rate of large phytoplankton
                ExcPLn = p_gl*GppPLn
                # (9) Grazing rate of small phytoplankton by small zooplankton
                GraPS2ZSn = max(0.0, p_grmaxsps*np.exp(p_kgras*temp_c)
                    *(1.0 - np.exp(p_ls*(p_ps2zs - PSn_c))) * ZSn_c)
                # (10) Grazing rate of small phytoplankton by large zooplankton
                GraPS2ZLn = max(0.0, p_grmaxlps*np.exp(p_kgral*temp_c)
                    *(1.0 - np.exp(p_ll*(p_ps2zl - ZSn_c))) * ZLn_c)
                # (11) Grazing rate of large phytoplankton by large zooplankton
                GraPL2ZLn = max(0.0, p_grmaxlpl*np.exp(p_kgral*temp_c)
                    *(1.0 - np.exp(p_ll*(p_pl2zl - PLn_c))) * ZLn_c)
                # (12) Grazing rate of small zooplankton by large zooplankton
                GraZS2ZLn = max(0.0, p_grmaxlzs*np.exp(p_kgral*temp_c)
                    *(1.0 - np.exp(p_ll*(p_zs2zl - ZSn_c))) * ZLn_c)
                # (13) Grazing rate of large phytoplankton by predatory zooplankton
                GraPL2ZPn = max(0.0, p_grmaxppl*np.exp(p_kgrap*temp_c)
                    *(1.0 - np.exp(p_lp*(p_pl2zp - PLn_c))) * ZPn_c
                    *np.exp(-p_thipl*(ZLn_c + ZSn_c)))
                # (14) Grazing rate of small zooplankton by predatory zooplankton
                GraZS2ZPn = max(0.0, p_grmaxpzs*np.exp(p_kgrap*temp_c)
                    *(1.0 - np.exp(p_lp*(p_zs2zp - ZSn_c))) * ZPn_c
                    *np.exp(-p_thizs*ZLn_c))
                # (15) Grazing rate of large zooplankton by predatory zooplankton
                GraZL2ZPn = max(0.0, p_grmaxpzl*np.exp(p_kgrap*temp_c)
                    *(1.0 - np.exp(p_lp*(p_zl2zp - ZLn_c))) * ZPn_c)
                # (16) Excretion rate of small zooplankton
                ExcZSn = (p_alphazs - p_betazs)*GraPS2ZSn
                # (17) Excretion rate of large zooplankton
                ExcZLn = (p_alphazl - p_betazl
                    )*(GraPL2ZLn + GraZS2ZLn + GraPS2ZLn)
                # (18) Excretion rate of predatory zooplankton
                ExcZPn = (p_alphazp - p_betazp
                    )*(GraPL2ZPn + GraZS2ZPn + GraPL2ZPn) # Note: GraPL2ZPn appears twice here in original, check if intended
                # (19) Egestion rate of small zooplankton
                EgeZSn = (1.0 - p_alphazs) * GraPS2ZSn
                # (20) Egestion rate of large zooplankton
                EgeZLn = (1.0 - p_alphazl) * (
                    GraPL2ZLn + GraZS2ZLn + GraPS2ZLn)
                # (21) Egestion rate of predatory zooplankton
                EgeZPn = (1.0 - p_alphazp) * (
                    GraPL2ZPn + GraZS2ZPn + GraZL2ZPn)
                # (22) Mortality rate of small zooplankton
                MorZSn = p_morzs0 * np.exp(p_kmorzs * temp_c) * ZSn_c**2
                # (23) Mortality rate of large zooplankton
                MorZLn = p_morzl0 * np.exp(p_kmorzl * temp_c) * ZLn_c**2
                # (24) Mortality rate of predatory zooplankton
                MorZPn = p_morzp0 * np.exp(p_kmorzp * temp_c) * ZPn_c**2
                # (25) Decomposition rate from PON to ammonium
                DecP2N = p_vp2n0 * np.exp(p_kp2n * temp_c) * PON_c 
                # (26) Decomposition rate from PON to DON
                DecP2D = p_vp2d0 * np.exp(p_kp2d * temp_c) * PON_c 
                # (27) Decomposition rate from DON to ammonium
                DecD2N = p_vd2n0 * np.exp(p_kd2n * temp_c) * DON_c 
                # (28) Nitrification rate
                Nit = p_nit0 * np.exp(p_knit * temp_c) * ammonium_c
                # (29) Sinking rate of PON
                ### Computed elsewhere
                
                # A.4. Silicon

                # (30) Gross primary production rate of large phytoplankton
                GppPLsi = GppPLn * p_rsinpl
                # (31) Respiration rate of large 
                ResPLsi = ResPLn * p_rsinpl
                # (32) Mortality rate of large phytoplankton
                MorPLsi = MorPLn * p_rsinpl
                # (33) Extracellular excretion rate of large phytoplankton
                ExcPLsi = ExcPLn * p_rsinpl
                # (34) Grazing rate of large phytoplankton by large zooplankton
                GraPL2ZLsi = GraPL2ZLn * p_rsinpl
                # (35) Grazing rate of large phytoplankton by predatory zooplankton
                GraPL2ZPsi = GraPL2ZPn * p_rsinpl
                # (36) Egestion rate of large zooplankton
                EgeZLsi = EgeZLn * p_rsinpl
                # (37) Egestion rate of predatory zooplankton
                EgeZPsi = EgeZPn * p_rsinpl
                # (38) Decomposition rate from Opal to silicate
                DecP2Si = p_vp2si0 * np.exp(p_kp2si * temp_c) * Opal_c
                # (39) Sedimentation rate of Opal [sinking?]
                ### Computed elsewhere
            
                # --- E. DERIVATIVES ---
                
                # A.1. Nitrogen
                d_PSn = GppPSn - ResPSn - MorPSn - ExcPSn - GraPS2ZSn - GraPS2ZLn
                d_PLn = GppPLn - ResPLn - MorPLn - ExcPLn - GraPL2ZLn - GraPL2ZPn
                d_ZSn = GraPS2ZSn - GraZS2ZLn - GraZS2ZPn - MorZSn - ExcZSn - EgeZSn
                d_ZLn = GraPS2ZLn + GraPL2ZLn + GraZS2ZLn - GraZL2ZPn - MorZLn - ExcZLn - EgeZLn
                d_ZPn = GraPL2ZPn + GraZS2ZPn + GraZL2ZPn - MorZPn - ExcZPn - EgeZPn
                d_nitrate = - (GppPSn - ResPSn)*RnewS  - (GppPLn - ResPLn)*RnewL + Nit
                d_ammonium = - (GppPSn - ResPSn)*(1.0 - RnewS) - (GppPLn - ResPLn)*(1.0 - RnewL) - Nit + DecP2N + DecD2N + ExcZSn + ExcZLn + ExcZPn
                d_PON = MorPSn + MorPLn + MorZSn + MorZLn  + MorZPn + EgeZSn + EgeZLn + EgeZPn - DecP2N - DecP2D
                
                # A.2. Silicon
                d_DON = ExcPSn + ExcPLn + DecP2D - DecD2N
                d_PLsi = GppPLsi - ResPLsi - MorPLsi - ExcPLsi - GraPL2ZLsi - GraPL2ZPsi
                d_ZLsi = GraPL2ZLsi - EgeZLsi
                d_ZPsi = GraPL2ZPsi - EgeZPsi
                d_silicate = - GppPLsi + ResPLsi + ExcPLsi + DecP2Si
                d_Opal = MorPLsi + EgeZLsi + EgeZPsi - DecP2Si

                # Chlorophyll-a
                d_schl = p_thetas * GppPSn**2 / (PSn_c + 1e-12) / (p_alphas * par_c + 1e-12) - (ResPSn + MorPSn + ExcPSn + GraPS2ZSn + GraPS2ZLn) / (PSn_c + 1e-12) * schl_c
                d_lchl = p_thetal * GppPLn**2 / (PLn_c + 1e-12) / (p_alphal * par_c + 1e-12) - (ResPLn + MorPLn + ExcPLn + GraPL2ZLn + GraPL2ZPn) / (PLn_c + 1e-12) * lchl_c                    
                
                d_oxygen = - 172.0 / 16.0 * (d_nitrate + d_ammonium + Nit)
                
                # --- F. UPDATE ---
                PSn_n[k, j, i] = PSn_c + d_PSn * dt_day
                PLn_n[k, j, i] = PLn_c + d_PLn * dt_day
                ZSn_n[k, j, i] = ZSn_c + d_ZSn * dt_day
                ZLn_n[k, j, i] = ZLn_c + d_ZLn * dt_day
                ZPn_n[k, j, i] = ZPn_c + d_ZPn * dt_day
                nitrate_n[k, j, i] = nitrate_c + d_nitrate * dt_day
                ammonium_n[k, j, i] = ammonium_c + d_ammonium * dt_day
                PON_n[k, j, i] = PON_c + d_PON * dt_day
                DON_n[k, j, i] = DON_c + d_DON * dt_day
                PLsi_n[k, j, i] = PLsi_c + d_PLsi * dt_day
                ZLsi_n[k, j, i] = ZLsi_c + d_ZLsi * dt_day
                ZPsi_n[k, j, i] = ZPsi_c + d_ZPsi * dt_day
                silicate_n[k, j, i] = silicate_c + d_silicate * dt_day
                Opal_n[k, j, i] = Opal_c + d_Opal * dt_day
                schl_n[k, j, i] = schl_c + d_schl * dt_day
                lchl_n[k, j, i] = lchl_c + d_lchl * dt_day                    
                oxygen_n[k, j, i] = oxygen_c + d_oxygen * dt_day
        
        return PSn_n, PLn_n, ZSn_n, ZLn_n, ZPn_n, nitrate_n, ammonium_n, PON_n, DON_n, PLsi_n, ZLsi_n, ZPsi_n, silicate_n, Opal_n, schl_n, lchl_n, oxygen_n

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