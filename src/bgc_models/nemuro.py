import taichi as ti
import numpy as np
from dataclasses import dataclass
from .base import BaseBGCModel
from .utils import calculate_par_ti, apply_sinking_ti

@ti.kernel
def nemuro_kernel_ti(
    PSn: ti.types.ndarray(), PLn: ti.types.ndarray(), ZSn: ti.types.ndarray(),
    ZLn: ti.types.ndarray(), ZPn: ti.types.ndarray(), nitrate: ti.types.ndarray(),
    ammonium: ti.types.ndarray(), PON: ti.types.ndarray(), DON: ti.types.ndarray(),
    PLsi: ti.types.ndarray(), ZLsi: ti.types.ndarray(), ZPsi: ti.types.ndarray(),
    silicate: ti.types.ndarray(), Opal: ti.types.ndarray(), schl: ti.types.ndarray(),
    lchl: ti.types.ndarray(), oxygen: ti.types.ndarray(),
    temp: ti.types.ndarray(), par: ti.types.ndarray(), dz: ti.types.ndarray(),
    dt: ti.f32, p: ti.types.ndarray()
):
    dt_day = dt / 86400.0
    cn_mass = (106.0 / 16.0) * 12.01
    nz, ny, nx = PSn.shape

    # --- A. Unpack Parameters ---
    p_vmaxs = p[0]; p_kno3s = p[1]; p_knh4s = p[2]; p_this = p[3]; p_kgpps = p[4]
    p_iopts = p[5]; p_resps0 = p[6]; p_kresps = p[7]; p_gs = p[8]; p_morps0 = p[9]
    p_kmorps = p[10]; p_vmaxl = p[11]; p_kno3l = p[12]; p_knh4l = p[13]; p_ksil = p[14]
    p_thil = p[15]; p_kgppl = p[16]; p_ioptl = p[17]; p_respl0 = p[18]; p_krespl = p[19]
    p_gl = p[20]; p_morpl0 = p[21]; p_kmorpl = p[22]; p_grmaxsps = p[23]; p_kgras = p[24]
    p_ls = p[25]; p_ps2zs = p[26]; p_morzs0 = p[27]; p_kmorzs = p[28]; p_alphazs = p[29]
    p_betazs = p[30]; p_grmaxlps = p[31]; p_grmaxlpl = p[32]; p_grmaxlzs = p[33]; p_kgral = p[34]
    p_ll = p[35]; p_ps2zl = p[36]; p_pl2zl = p[37]; p_zs2zl = p[38]; p_morzl0 = p[39]
    p_kmorzl = p[40]; p_alphazl = p[41]; p_betazl = p[42]; p_grmaxppl = p[43]; p_grmaxpzs = p[44]
    p_grmaxpzl = p[45]; p_kgrap = p[46]; p_lp = p[47]; p_pl2zp = p[48]; p_zs2zp = p[49]
    p_zl2zp = p[50]; p_thipl = p[51]; p_thizs = p[52]; p_morzp0 = p[53]; p_kmorzp = p[54]
    p_alphazp = p[55]; p_betazp = p[56]; p_nit0 = p[57]; p_knit = p[58]; p_vp2n0 = p[59]
    p_kp2n = p[60]; p_vp2d0 = p[61]; p_kp2d = p[62]; p_vd2n0 = p[63]; p_kd2n = p[64]
    p_vp2si0 = p[65]; p_kp2si = p[66]; p_rsinpl = p[67]; p_setvp = p[68]; p_setvo = p[69]
    p_thetas = p[70]; p_thetal = p[71]; p_alphas = p[72]; p_alphal = p[73]

    # --- B. GPU Multi-Threading Loop ---
    # Parallels seamlessly across all ocean cells simultaneously
    for k, j, i in ti.ndrange(nz, ny, nx):
        if dz[k, j, i] > 1e-6:
            
            # Read current values
            c_PSn = PSn[k, j, i]; c_PLn = PLn[k, j, i]; c_ZSn = ZSn[k, j, i]
            c_ZLn = ZLn[k, j, i]; c_ZPn = ZPn[k, j, i]; c_nitrate = nitrate[k, j, i]
            c_ammonium = ammonium[k, j, i]; c_PON = PON[k, j, i]; c_DON = DON[k, j, i]
            c_PLsi = PLsi[k, j, i]; c_ZLsi = ZLsi[k, j, i]; c_ZPsi = ZPsi[k, j, i]
            c_silicate = silicate[k, j, i]; c_Opal = Opal[k, j, i]
            c_schl = schl[k, j, i]; c_lchl = lchl[k, j, i]; c_oxygen = oxygen[k, j, i]
            
            c_temp = temp[k, j, i]; c_par = par[k, j, i]

            # (1) GPP Small Phytoplankton
            n_lim_s = (c_nitrate/(c_nitrate + p_kno3s) * ti.math.exp(-p_this * c_ammonium) 
                       + c_ammonium/(c_ammonium + p_knh4s))
            GppPSn = p_vmaxs * n_lim_s * ti.math.exp(p_kgpps * c_temp) * (c_par/p_iopts) * ti.math.exp(1.0 - c_par/p_iopts) * c_PSn
            RnewS = (c_nitrate/(c_nitrate + p_kno3s) * ti.math.exp(-p_this*c_ammonium)) / (n_lim_s + 1e-12)

            # (2) GPP Large Phytoplankton
            n_lim_l = (c_nitrate/(c_nitrate+p_kno3l)*ti.math.exp(-p_thil*c_ammonium) + c_ammonium/(c_ammonium+p_knh4l))
            Si_lim = c_silicate/(c_silicate+p_ksil)/p_rsinpl
            GppPLn = p_vmaxl * ti.min(n_lim_l, Si_lim) * ti.math.exp(p_kgppl*c_temp) * (c_par/p_ioptl) * ti.math.exp(1.0 - c_par/p_ioptl) * c_PLn
            RnewL = (c_nitrate/(c_nitrate + p_kno3l) * ti.math.exp(-p_thil*c_ammonium)) / (n_lim_l + 1e-12)

            # (3-8) Respiration, Mortality, Excretion
            ResPSn = p_resps0 * ti.math.exp(p_kresps * c_temp) * c_PSn
            ResPLn = p_respl0 * ti.math.exp(p_krespl * c_temp) * c_PLn
            MorPSn = p_morps0 * ti.math.exp(p_kmorps * c_temp) * c_PSn**2
            MorPLn = p_morpl0 * ti.math.exp(p_kmorpl * c_temp) * c_PLn**2
            ExcPSn = p_gs * GppPSn
            ExcPLn = p_gl * GppPLn

            # (9-15) Grazing Rates
            GraPS2ZSn = ti.max(0.0, p_grmaxsps * ti.math.exp(p_kgras * c_temp) * (1.0 - ti.math.exp(p_ls*(p_ps2zs - c_PSn))) * c_ZSn)
            GraPS2ZLn = ti.max(0.0, p_grmaxlps * ti.math.exp(p_kgral * c_temp) * (1.0 - ti.math.exp(p_ll*(p_ps2zl - c_ZSn))) * c_ZLn)
            GraPL2ZLn = ti.max(0.0, p_grmaxlpl * ti.math.exp(p_kgral * c_temp) * (1.0 - ti.math.exp(p_ll*(p_pl2zl - c_PLn))) * c_ZLn)
            GraZS2ZLn = ti.max(0.0, p_grmaxlzs * ti.math.exp(p_kgral * c_temp) * (1.0 - ti.math.exp(p_ll*(p_zs2zl - c_ZSn))) * c_ZLn)
            
            GraPL2ZPn = ti.max(0.0, p_grmaxppl * ti.math.exp(p_kgrap * c_temp) * (1.0 - ti.math.exp(p_lp*(p_pl2zp - c_PLn))) * c_ZPn * ti.math.exp(-p_thipl*(c_ZLn + c_ZSn)))
            GraZS2ZPn = ti.max(0.0, p_grmaxpzs * ti.math.exp(p_kgrap * c_temp) * (1.0 - ti.math.exp(p_lp*(p_zs2zp - c_ZSn))) * c_ZPn * ti.math.exp(-p_thizs*c_ZLn))
            GraZL2ZPn = ti.max(0.0, p_grmaxpzl * ti.math.exp(p_kgrap * c_temp) * (1.0 - ti.math.exp(p_lp*(p_zl2zp - c_ZLn))) * c_ZPn)

            # (16-24) Zooplankton Excretion, Egestion, Mortality
            ExcZSn = (p_alphazs - p_betazs) * GraPS2ZSn
            ExcZLn = (p_alphazl - p_betazl) * (GraPL2ZLn + GraZS2ZLn + GraPS2ZLn)
            ExcZPn = (p_alphazp - p_betazp) * (GraPL2ZPn + GraZS2ZPn + GraZL2ZPn)
            
            EgeZSn = (1.0 - p_alphazs) * GraPS2ZSn
            EgeZLn = (1.0 - p_alphazl) * (GraPL2ZLn + GraZS2ZLn + GraPS2ZLn)
            EgeZPn = (1.0 - p_alphazp) * (GraPL2ZPn + GraZS2ZPn + GraZL2ZPn)
            
            MorZSn = p_morzs0 * ti.math.exp(p_kmorzs * c_temp) * c_ZSn**2
            MorZLn = p_morzl0 * ti.math.exp(p_kmorzl * c_temp) * c_ZLn**2
            MorZPn = p_morzp0 * ti.math.exp(p_kmorzp * c_temp) * c_ZPn**2

            # (25-28) Decomposition and Nitrification
            DecP2N = p_vp2n0 * ti.math.exp(p_kp2n * c_temp) * c_PON 
            DecP2D = p_vp2d0 * ti.math.exp(p_kp2d * c_temp) * c_PON 
            DecD2N = p_vd2n0 * ti.math.exp(p_kd2n * c_temp) * c_DON 
            Nit    = p_nit0 * ti.math.exp(p_knit * c_temp) * c_ammonium

            # (30-38) Silicon Cycle
            GppPLsi = GppPLn * p_rsinpl; ResPLsi = ResPLn * p_rsinpl
            MorPLsi = MorPLn * p_rsinpl; ExcPLsi = ExcPLn * p_rsinpl
            GraPL2ZLsi = GraPL2ZLn * p_rsinpl; GraPL2ZPsi = GraPL2ZPn * p_rsinpl
            EgeZLsi = EgeZLn * p_rsinpl; EgeZPsi = EgeZPn * p_rsinpl
            DecP2Si = p_vp2si0 * ti.math.exp(p_kp2si * c_temp) * c_Opal

            # --- C. DERIVATIVES ---
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

            d_schl = p_thetas * GppPSn**2 / (c_PSn + 1e-12) / (p_alphas * c_par + 1e-12) * cn_mass - (ResPSn + MorPSn + ExcPSn + GraPS2ZSn + GraPS2ZLn) / (c_PSn + 1e-12) * c_schl      
            d_lchl = p_thetal * GppPLn**2 / (c_PLn + 1e-12) / (p_alphal * c_par + 1e-12) * cn_mass - (ResPLn + MorPLn + ExcPLn + GraPL2ZLn + GraPL2ZPn) / (c_PLn + 1e-12) * c_lchl      
            d_oxygen = - (172.0 / 16.0) * (d_nitrate + d_ammonium + Nit)

            # --- D. UPDATE IN-PLACE (With safe bounds) ---
            # Modifies the arrays directly in memory! No return statements required.
            PSn[k, j, i] = ti.max(0.0, c_PSn + d_PSn * dt_day)
            PLn[k, j, i] = ti.max(0.0, c_PLn + d_PLn * dt_day)
            ZSn[k, j, i] = ti.max(0.0, c_ZSn + d_ZSn * dt_day)
            ZLn[k, j, i] = ti.max(0.0, c_ZLn + d_ZLn * dt_day)
            ZPn[k, j, i] = ti.max(0.0, c_ZPn + d_ZPn * dt_day)
            nitrate[k, j, i] = ti.max(0.0, c_nitrate + d_nitrate * dt_day)
            ammonium[k, j, i] = ti.max(0.0, c_ammonium + d_ammonium * dt_day)
            PON[k, j, i] = ti.max(0.0, c_PON + d_PON * dt_day)
            DON[k, j, i] = ti.max(0.0, c_DON + d_DON * dt_day)
            PLsi[k, j, i] = ti.max(0.0, c_PLsi + d_PLsi * dt_day)
            ZLsi[k, j, i] = ti.max(0.0, c_ZLsi + d_ZLsi * dt_day)
            ZPsi[k, j, i] = ti.max(0.0, c_ZPsi + d_ZPsi * dt_day)
            silicate[k, j, i] = ti.max(0.0, c_silicate + d_silicate * dt_day)
            Opal[k, j, i] = ti.max(0.0, c_Opal + d_Opal * dt_day)
            schl[k, j, i] = ti.max(0.0, c_schl + d_schl * dt_day)
            lchl[k, j, i] = ti.max(0.0, c_lchl + d_lchl * dt_day)                  
            oxygen[k, j, i] = ti.max(0.0, c_oxygen + d_oxygen * dt_day)


class Model_NEMURO(BaseBGCModel):
    def __init__(self, nz, ny, nx, water_mask, params=None):
        super().__init__(nz, ny, nx, water_mask)
        
        self.names = [
            'PSn', 'PLn', 'ZSn', 'ZLn', 'ZPn', 'nitrate', 'ammonium', 'PON', 'DON', 
            'PLsi', 'ZLsi', 'ZPsi', 'silicate', 'Opal', 'schl', 'lchl', 'oxygen'
        ]
        # Keep tracers as NumPy arrays for I/O. Taichi will zero-copy access them!
        self.tracers = {n: np.zeros((nz, ny, nx), dtype=np.float32) for n in self.names}
        self.par_3d = np.zeros((nz, ny, nx), dtype=np.float32)
        
        self.params = params if params is not None else Params_BGC()
        
        # Pack the 74 parameters into a fast, 1D NumPy array for the GPU
        param_order = [
            'p_vmaxs', 'p_kno3s', 'p_knh4s', 'p_this', 'p_kgpps', 'p_iopts', 'p_resps0', 'p_kresps', 'p_gs', 'p_morps0',
            'p_kmorps', 'p_vmaxl', 'p_kno3l', 'p_knh4l', 'p_ksil', 'p_thil', 'p_kgppl', 'p_ioptl', 'p_respl0', 'p_krespl',
            'p_gl', 'p_morpl0', 'p_kmorpl', 'p_grmaxsps', 'p_kgras', 'p_ls', 'p_ps2zs', 'p_morzs0', 'p_kmorzs', 'p_alphazs',
            'p_betazs', 'p_grmaxlps', 'p_grmaxlpl', 'p_grmaxlzs', 'p_kgral', 'p_ll', 'p_ps2zl', 'p_pl2zl', 'p_zs2zl', 'p_morzl0',
            'p_kmorzl', 'p_alphazl', 'p_betazl', 'p_grmaxppl', 'p_grmaxpzs', 'p_grmaxpzl', 'p_kgrap', 'p_lp', 'p_pl2zp', 'p_zs2zp',
            'p_zl2zp', 'p_thipl', 'p_thizs', 'p_morzp0', 'p_kmorzp', 'p_alphazp', 'p_betazp', 'p_nit0', 'p_knit', 'p_vp2n0',
            'p_kp2n', 'p_vp2d0', 'p_kp2d', 'p_vd2n0', 'p_kd2n', 'p_vp2si0', 'p_kp2si', 'p_rsinpl', 'p_setvp', 'p_setvo',
            'p_thetas', 'p_thetal', 'p_alphas', 'p_alphal'
        ]
        self.param_arr = np.array([getattr(self.params, name) for name in param_order], dtype=np.float32)

        self.sinking_config = {
            'PON': self.params.p_setvp,
            'Opal': self.params.p_setvo
        }  

    def biology_step(self, t_curr, sw_curr, dz, dt):
        """Step biology forward directly on the GPU."""
        
        # 1. Calc Shared PAR (In-place update of self.par_3d)
        psum = self.tracers['schl'] + self.tracers['lchl']
        self.par_3d.fill(0.0)
        calculate_par_ti(sw_curr.astype(np.float32), psum, dz.astype(np.float32), self.par_3d)

        # 2. Call the Taichi GPU Kernel 
        nemuro_kernel_ti(
            self.tracers['PSn'], self.tracers['PLn'], self.tracers['ZSn'],
            self.tracers['ZLn'], self.tracers['ZPn'], self.tracers['nitrate'],
            self.tracers['ammonium'], self.tracers['PON'], self.tracers['DON'],
            self.tracers['PLsi'], self.tracers['ZLsi'], self.tracers['ZPsi'],
            self.tracers['silicate'], self.tracers['Opal'], self.tracers['schl'],
            self.tracers['lchl'], self.tracers['oxygen'],
            t_curr.astype(np.float32), self.par_3d, dz.astype(np.float32), 
            float(dt), self.param_arr
        )
        
        # No return dictionary needed! All 17 arrays were updated in-place.
        return self.par_3d

    def sinking_step(self, dz, dt):
        # Find minimum safe cell depth for the sub-stepping limits
        dz_safe_min = np.where(dz > 1e-6, dz, 1e6)
        min_dz = float(np.min(dz_safe_min))
        
        for var, w_sink in self.sinking_config.items():
            if w_sink > 0.0:
                apply_sinking_ti(self.tracers[var], dz.astype(np.float32), float(dt), float(w_sink), min_dz)


@dataclass
class Params_BGC:
    # Default values from Table 1 Station A of Yoshie et al. 2007.
    p_vmaxs: float = 0.4
    p_kno3s: float = 1.0
    p_knh4s: float = 0.1
    p_this: float = 1.5
    p_kgpps: float = 0.0693
    p_iopts: float = 104.7
    p_resps0: float = 0.03
    p_kresps: float = 0.0519
    p_gs: float = 0.135
    p_morps0: float = 0.0585
    p_kmorps: float = 0.0693
    p_vmaxl: float = 0.8
    p_kno3l: float = 3.0
    p_knh4l: float = 0.3
    p_ksil: float = 6.0
    p_thil: float = 1.5
    p_kgppl: float = 0.0693
    p_ioptl: float = 104.7
    p_respl0: float = 0.03
    p_krespl: float = 0.0519
    p_gl: float = 0.135
    p_morpl0: float = 0.029
    p_kmorpl: float = 0.0693
    p_grmaxsps: float = 0.4
    p_kgras: float = 0.0693
    p_ls: float = 1.4
    p_ps2zs: float = 0.04
    p_morzs0: float = 0.0585
    p_kmorzs: float = 0.0693
    p_alphazs: float = 0.7
    p_betazs: float = 0.3
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
    p_rsinpl: float = 2.0
    p_setvp: float = 40
    p_setvo: float = 40
    p_thetas: float = 0.0328
    p_thetal: float = 0.0386
    p_alphas: float = 0.0405
    p_alphal: float = 0.0393