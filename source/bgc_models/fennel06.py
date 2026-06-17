import numpy as np
from dataclasses import dataclass
from numba import njit, prange
from .base import BaseBGCModel
from .utils import calculate_par

"""
Fennel et al. (2006): Nitrogen cycling in the MidAtlantic Bight and implications for the North Atlantic nitrogen budget: Results from a three-dimensional model. Global Biogeochemical Cycles 20, GB3007, doi:10.1029/2005GB002456

For code adaptation, I followed https://github.com/bwang63/gotm-fabm-memg-biogeochemical-model/blob/main/bio_fennel/1p1z.F90
"""

class Model_FENNEL06(BaseBGCModel):
    def __init__(self, nz, ny, nx, water_mask, params=None):
        super().__init__(nz, ny, nx, water_mask)
        
        # 1. Setup Tracers
        self.names = [
            'nitrate', 
            'ammonium', 
            'phy',
            'chl',
            'zoo', 
            'sdet',
            'ldet', 
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
            'phy': self.params.p_wp,
            'sdet': self.params.p_ws,
            'ldet': self.params.p_wl
        }

    def biology_step(self, t_curr, sw_curr, dz, dt):
        """Step biology forward."""
        # 1. Calc Shared PAR
        psum = self.tracers['chl']
        par_3d = calculate_par(sw_curr, psum, dz)

        # 2. Unpack Params for Numba 
        # (Numba cannot read self.params, we must pass floats)
        p = self.params
        
        # 2. Call Specific Kernel
        ( 
         self.tracers['nitrate'], 
         self.tracers['ammonium'], 
         self.tracers['phy'], 
         self.tracers['chl'], 
         self.tracers['zoo'],
         self.tracers['sdet'],
         self.tracers['ldet'],
         self.tracers['oxygen']
        ) = self._run_kernel(
            self.tracers['nitrate'], 
            self.tracers['ammonium'], 
            self.tracers['phy'], 
            self.tracers['chl'], 
            self.tracers['zoo'],
            self.tracers['sdet'],
            self.tracers['ldet'],
            self.tracers['oxygen'],
            t_curr, par_3d, dz, dt,
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
        return par_3d
    

    @staticmethod
    @njit(parallel=True, fastmath=True)
    def _run_kernel(
        nitrate, ammonium, phy, chl, zoo, sdet, ldet, oxygen,
        temp, par, dz, dt,
        # Parameters as arguments
        p_i_thnh4, p_d_p5nh4, p_nitrir, p_k_no3, p_k_nh4, p_vp0,
        p_k_phy, p_phycn, p_phyis, p_phymr, p_chl2c_m, p_zooae_n,
        p_zoobm, p_zooer, p_zoogr, p_zoomr, p_zoocn, p_lderrn,
        p_sderrn, p_coagr, p_wp, p_ws, p_wl
    ):

        nz, ny, nx = nitrate.shape
        
        # Time conversion (dt expressed in day)
        dt_day = dt / 86400.0
        eps = 1e-12
        
        # Initialize Outputs (_n for next time step)
        # Using .copy() instead of zeros_like to preserve masked/land values untouched
        nitrate_n = nitrate.copy()
        ammonium_n = ammonium.copy()
        phy_n = phy.copy()
        chl_n = chl.copy()
        zoo_n = zoo.copy()
        sdet_n = sdet.copy()
        ldet_n = ldet.copy()
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
                    
                # Load State (_c for current time step)
                nitrate_c = nitrate[k, j, i]
                ammonium_c = ammonium[k, j, i]
                phy_c = phy[k, j, i]
                chl_c = chl[k, j, i]
                zoo_c = zoo[k, j, i]
                sdet_c = sdet[k, j, i]
                ldet_c = ldet[k, j, i]
                oxygen_c = oxygen[k, j, i]                
                
                temp_c = temp[k, j, i]
                par_c = par[k, j, i]                
                
                # --- D. SOURCES / SINKS ---
                
                # chl-to-carbon ratio
                cff = p_phycn * 12.0
                chl2c = min(chl_c/(phy_c*cff+eps), p_chl2c_m)

                # temperature- and light-limited growth rate (Eppley 1972)
                vp = p_vp0 * 0.59 * (1.066 ** temp_c)
                fac1 = par_c * p_phyis
                epp = vp / np.sqrt(vp*vp + fac1*fac1)
                t_ppmax = epp * fac1

                # nutrient-limitation (Parker 1993)
                cff1 = ammonium_c * p_k_nh4
                cff2 = nitrate_c * p_k_no3
                inhnh4 = 1.0 / (1.0 + cff1)
                l_nh4 = cff1 / (1.0 + cff1)
                l_no3 = cff2 * inhnh4 / (1.0 + cff2)
                ltot = l_no3 + l_nh4

                # nitrate and ammonium uptake by phytoplankton
                fac1 = t_ppmax 
                cff4 = fac1 * p_k_no3 * inhnh4 / (1.0 + cff2) * phy_c
                cff5 = fac1 * p_k_nh4 / (1.0 + cff1) * phy_c

                n_newprod = nitrate_c * cff4
                n_regprod = ammonium_c * cff5

                chl_prod = (t_ppmax*t_ppmax*ltot*ltot*p_chl2c_m*chl_c
                           ) / (p_phyis * max(chl2c, eps) * par_c + eps)

                # nitrification (nh4 --> no3, Olson 1981)
                fac2 = max(oxygen_c, 0.0)
                fac3 = max(fac2 / (3.0 + fac2), 0.0)
                fac1 = p_nitrir * fac3

                cff1 = (par_c - p_i_thnh4) / (p_d_p5nh4 + par_c - 2.0 * p_i_thnh4)
                cff2 = 1.0 - max(cff1, 0.0)
                cff3 = fac1 * cff2
                n_nitrifi = ammonium_c * cff3

                # if no light, no phyto growth and nitrification occurs at max rate
                if par_c == 0.0:
                    n_newprod = 0.0
                    n_regprod = 0.0
                    chl_prod = 0.0

                # zooplankton grazing (Landry 1993)
                fac1 = p_zoogr
                cff2 = p_phymr
                cff1 = fac1 * zoo_c * phy_c / (p_k_phy + phy_c * phy_c)
                n_graz = cff1 * phy_c
                chl_graz = cff1 * chl_c

                n_assim = cff1 * phy_c * p_zooae_n
                n_egest = cff1 * phy_c * (1.0 - p_zooae_n)
                
                # phyto mortality
                n_pmortal = cff2 * max(phy_c - 1e-6, 0.0)
                chl_pmortal = cff2 * max(chl_c - 1e-6, 0.0)

                # zoo loss terms
                cff1 = p_zoobm
                fac2 = p_zoomr
                fac3 = p_zooer
                fac1 = fac3 * phy_c * phy_c / (p_k_phy + phy_c * phy_c)
                cff2 = fac2 * zoo_c
                cff3 = fac1 * p_zooae_n

                n_zmortal = cff2 * zoo_c
                n_zexcret = cff3 * zoo_c
                n_zmetabo = cff1 * max(zoo_c - 1e-6, 0.0)

                # coagulation of phy and sdet to ldet
                fac1 = p_coagr
                cff1 = fac1 * (sdet_c + phy_c)
                n_coagp = phy_c * cff1
                chl_coag = chl_c * cff1
                n_coagd = sdet_c * cff1

                # remineralization
                fac1 = max(oxygen_c - 6.0, 0.0)
                fac2 = max(fac1 / (3.0 + fac1), 0.0)
                cff1 = p_sderrn * fac2
                cff3 = p_lderrn * fac2
                n_remines = sdet_c * cff1
                n_reminel = ldet_c * cff3

                # --- E. DERIVATIVES ---
                
                d_nitrate = - n_newprod + n_nitrifi
                d_ammonium = - n_regprod - n_nitrifi + n_zmetabo + n_zexcret + n_remines + n_reminel
                d_phy = n_newprod + n_regprod - n_graz - n_pmortal - n_coagp
                d_chl = chl_prod - chl_graz - chl_pmortal - chl_coag
                d_zoo = n_assim - n_zmortal - n_zexcret - n_zmetabo
                d_sdet = n_egest + n_pmortal + n_zmortal - n_coagd - n_remines
                d_ldet = n_coagp + n_coagd - n_reminel
                
                # Combining the last two terms given the same C:N ratio as suggested in your comment
                d_oxygen = n_newprod * 8.625 + n_regprod * 6.625 - 2.0 * n_nitrifi - 6.625 * (n_zmetabo + n_zexcret + n_remines + n_reminel) 
                
                # --- F. UPDATE ---
                nitrate_n[k, j, i] = nitrate_c + d_nitrate * dt_day
                ammonium_n[k, j, i] = ammonium_c + d_ammonium * dt_day
                phy_n[k, j, i] = phy_c + d_phy * dt_day
                chl_n[k, j, i] = chl_c + d_chl * dt_day
                zoo_n[k, j, i] = zoo_c + d_zoo * dt_day
                sdet_n[k, j, i] = sdet_c + d_sdet * dt_day
                ldet_n[k, j, i] = ldet_c + d_ldet * dt_day
                oxygen_n[k, j, i] = oxygen_c + d_oxygen * dt_day
        
        return nitrate_n, ammonium_n, phy_n, chl_n, zoo_n, sdet_n, ldet_n, oxygen_n

@dataclass
class Params_BGC:
    """
    Parameter set for Fennel06.
    Default values from https://github.com/bwang63/gotm-fabm-memg-biogeochemical-model/blob/main/bio_fennel/1p1z.F90#L160
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