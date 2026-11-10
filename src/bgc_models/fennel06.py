import taichi as ti
import numpy as np
from dataclasses import dataclass
from .base import BaseBGCModel
from .utils import calculate_par_ti, apply_sinking_ti

@ti.kernel
def fennel_kernel_ti(
    nitrate: ti.types.ndarray(), ammonium: ti.types.ndarray(), 
    phy: ti.types.ndarray(), chl: ti.types.ndarray(), 
    zoo: ti.types.ndarray(), sdet: ti.types.ndarray(), 
    ldet: ti.types.ndarray(), oxygen: ti.types.ndarray(),
    temp: ti.types.ndarray(), par: ti.types.ndarray(), 
    dz: ti.types.ndarray(), dt: ti.f32, p: ti.types.ndarray()
):
    dt_day = dt / 86400.0
    eps = 1e-12
    nz, ny, nx = nitrate.shape

    # --- A. Unpack Parameters ---
    p_i_thnh4 = p[0]; p_d_p5nh4 = p[1]; p_nitrir = p[2]
    p_k_no3 = p[3]; p_k_nh4 = p[4]; p_vp0 = p[5]
    p_k_phy = p[6]; p_phycn = p[7]; p_phyis = p[8]
    p_phymr = p[9]; p_chl2c_m = p[10]; p_zooae_n = p[11]
    p_zoobm = p[12]; p_zooer = p[13]; p_zoogr = p[14]
    p_zoomr = p[15]; p_zoocn = p[16]; p_lderrn = p[17]
    p_sderrn = p[18]; p_coagr = p[19]

    # --- B. GPU Multi-Threading Loop ---
    for k, j, i in ti.ndrange(nz, ny, nx):
        if dz[k, j, i] > 1e-6:
            
            # Read current values
            c_n = nitrate[k, j, i]; c_a = ammonium[k, j, i]
            c_p = phy[k, j, i]; c_c = chl[k, j, i]
            c_z = zoo[k, j, i]; c_sd = sdet[k, j, i]
            c_ld = ldet[k, j, i]; c_ox = oxygen[k, j, i]
            
            t = temp[k, j, i]; irrad = par[k, j, i]

            # --- C. SOURCES / SINKS ---
            
            # chl-to-carbon ratio
            cff = p_phycn * 12.0
            chl2c = ti.min(c_c / (c_p * cff + eps), p_chl2c_m)

            # temperature- and light-limited growth rate (Eppley 1972)
            vp = p_vp0 * 0.59 * (1.066 ** t)
            fac1_light = irrad * p_phyis
            epp = vp / ti.sqrt(vp * vp + fac1_light * fac1_light)
            t_ppmax = epp * fac1_light

            # nutrient-limitation (Parker 1993)
            cff1_nut = c_a * p_k_nh4
            cff2_nut = c_n * p_k_no3
            inhnh4 = 1.0 / (1.0 + cff1_nut)
            l_nh4 = cff1_nut / (1.0 + cff1_nut)
            l_no3 = cff2_nut * inhnh4 / (1.0 + cff2_nut)
            ltot = l_no3 + l_nh4

            # nitrate and ammonium uptake by phytoplankton
            cff4_up = t_ppmax * p_k_no3 * inhnh4 / (1.0 + cff2_nut) * c_p
            cff5_up = t_ppmax * p_k_nh4 / (1.0 + cff1_nut) * c_p

            n_newprod = c_n * cff4_up
            n_regprod = c_a * cff5_up

            chl_prod = (t_ppmax * t_ppmax * ltot * ltot * p_chl2c_m * c_c) / (p_phyis * ti.max(chl2c, eps) * irrad + eps)

            # nitrification (nh4 --> no3, Olson 1981)
            fac2_o2 = ti.max(c_ox, 0.0)
            fac3_o2 = ti.max(fac2_o2 / (3.0 + fac2_o2), 0.0)
            fac1_nit = p_nitrir * fac3_o2

            cff1_nit = (irrad - p_i_thnh4) / (p_d_p5nh4 + irrad - 2.0 * p_i_thnh4)
            cff2_nit = 1.0 - ti.max(cff1_nit, 0.0)
            cff3_nit = fac1_nit * cff2_nit
            n_nitrifi = c_a * cff3_nit

            # if no light, no phyto growth and nitrification occurs at max rate
            if irrad == 0.0:
                n_newprod = 0.0
                n_regprod = 0.0
                chl_prod = 0.0

            # zooplankton grazing (Landry 1993)
            cff1_graz = p_zoogr * c_z * c_p / (p_k_phy + c_p * c_p)
            n_graz = cff1_graz * c_p
            chl_graz = cff1_graz * c_c

            n_assim = cff1_graz * c_p * p_zooae_n
            n_egest = cff1_graz * c_p * (1.0 - p_zooae_n)

            # phyto mortality
            n_pmortal = p_phymr * ti.max(c_p - 1e-6, 0.0)
            chl_pmortal = p_phymr * ti.max(c_c - 1e-6, 0.0)

            # zoo loss terms
            fac1_z = p_zooer * c_p * c_p / (p_k_phy + c_p * c_p)
            n_zmortal = p_zoomr * c_z * c_z
            n_zexcret = (fac1_z * p_zooae_n) * c_z
            n_zmetabo = p_zoobm * ti.max(c_z - 1e-6, 0.0)

            # coagulation of phy and sdet to ldet
            cff1_coag = p_coagr * (c_sd + c_p)
            n_coagp = c_p * cff1_coag
            chl_coag = c_c * cff1_coag
            n_coagd = c_sd * cff1_coag

            # remineralization
            fac1_rem = ti.max(c_ox - 6.0, 0.0)
            fac2_rem = ti.max(fac1_rem / (3.0 + fac1_rem), 0.0)
            n_remines = c_sd * (p_sderrn * fac2_rem)
            n_reminel = c_ld * (p_lderrn * fac2_rem)

            # --- D. DERIVATIVES ---
            d_nitrate = -n_newprod + n_nitrifi
            d_ammonium = -n_regprod - n_nitrifi + n_zmetabo + n_zexcret + n_remines + n_reminel
            d_phy = n_newprod + n_regprod - n_graz - n_pmortal - n_coagp
            d_chl = chl_prod - chl_graz - chl_pmortal - chl_coag
            d_zoo = n_assim - n_zmortal - n_zexcret - n_zmetabo
            d_sdet = n_egest + n_pmortal + n_zmortal - n_coagd - n_remines
            d_ldet = n_coagp + n_coagd - n_reminel
            d_oxygen = n_newprod * 8.625 + n_regprod * 6.625 - 2.0 * n_nitrifi - 6.625 * (n_zmetabo + n_zexcret + n_remines + n_reminel)

            # --- E. UPDATE IN-PLACE ---
            nitrate[k, j, i] = ti.max(0.0, c_n + d_nitrate * dt_day)
            ammonium[k, j, i] = ti.max(0.0, c_a + d_ammonium * dt_day)
            phy[k, j, i] = ti.max(0.0, c_p + d_phy * dt_day)
            chl[k, j, i] = ti.max(0.0, c_c + d_chl * dt_day)
            zoo[k, j, i] = ti.max(0.0, c_z + d_zoo * dt_day)
            sdet[k, j, i] = ti.max(0.0, c_sd + d_sdet * dt_day)
            ldet[k, j, i] = ti.max(0.0, c_ld + d_ldet * dt_day)
            oxygen[k, j, i] = ti.max(0.0, c_ox + d_oxygen * dt_day)


class Model_FENNEL06(BaseBGCModel):
    def __init__(self, nz, ny, nx, water_mask, params=None):
        super().__init__(nz, ny, nx, water_mask)
        
        self.names = ['nitrate', 'ammonium', 'phy', 'chl', 'zoo', 'sdet', 'ldet', 'oxygen']
        
        # Keep tracers as NumPy arrays for I/O. Taichi will zero-copy access them.
        self.tracers = {n: np.zeros((nz, ny, nx), dtype=np.float32) for n in self.names}
        self.par_3d = np.zeros((nz, ny, nx), dtype=np.float32)
        
        self.params = params if params is not None else Params_BGC()
        
        # Pack the parameters into a fast, 1D NumPy array for the GPU
        param_order = [
            'p_i_thnh4', 'p_d_p5nh4', 'p_nitrir', 'p_k_no3', 'p_k_nh4', 'p_vp0',
            'p_k_phy', 'p_phycn', 'p_phyis', 'p_phymr', 'p_chl2c_m', 'p_zooae_n',
            'p_zoobm', 'p_zooer', 'p_zoogr', 'p_zoomr', 'p_zoocn', 'p_lderrn',
            'p_sderrn', 'p_coagr', 'p_wp', 'p_ws', 'p_wl'
        ]
        self.param_arr = np.array([getattr(self.params, name) for name in param_order], dtype=np.float32)

        self.sinking_config = {
            'phy': self.params.p_wp,
            'sdet': self.params.p_ws,
            'ldet': self.params.p_wl
        }

    def biology_step(self, t_curr, sw_curr, dz, dt):
        """Step biology forward directly on the GPU."""
        
        # 1. Calc Shared PAR (In-place update of self.par_3d)
        psum = self.tracers['chl']
        self.par_3d.fill(0.0)
        calculate_par_ti(sw_curr.astype(np.float32), psum, dz.astype(np.float32), self.par_3d)

        # 2. Call the Taichi GPU Kernel 
        fennel_kernel_ti(
            self.tracers['nitrate'], self.tracers['ammonium'], 
            self.tracers['phy'], self.tracers['chl'], 
            self.tracers['zoo'], self.tracers['sdet'], 
            self.tracers['ldet'], self.tracers['oxygen'],
            t_curr.astype(np.float32), self.par_3d, dz.astype(np.float32), 
            float(dt), self.param_arr
        )
        
        return self.par_3d

    def sinking_step(self, dz, dt):
        """Step particulate sinking forward."""
        dz_safe_min = np.where(dz > 1e-6, dz, 1e6)
        min_dz = float(np.min(dz_safe_min))
        
        for var, w_sink in self.sinking_config.items():
            if w_sink > 0.0:
                apply_sinking_ti(self.tracers[var], dz.astype(np.float32), float(dt), float(w_sink), min_dz)


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