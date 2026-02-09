import numpy as np
from numba import njit, prange


@njit(parallel=True, fastmath=True)
def calculate_par(nswr, psum, dz):
    """
    Calculates 3D Photosynthetically Active Radiation (PAR) field.
    Includes self-shading by phytoplankton.
    
    Inputs:
      nshr: 2D surface net shortwave radiation (W/m2)
      psum: 3D phytoplankton biomass, sum of all classes (mmol/m3)
      dz: 3D layer thicknesses (m)
      
    Returns:
      par_3d: 3D Average PAR in each layer (W/m2)
    """
    nz, ny, nx = dz.shape
    par_3d = np.zeros((nz, ny, nx))
    
    # Coefficients
    par_fraction = 0.43  # Fraction of Shortwave that is PAR (400-700nm)
    k_water      = 0.04  # Attenuation of pure water (1/m)
    k_chl        = 0.03  # Attenuation per unit Phyto (m2/mmol) (Self-shading)
    
    for j in prange(ny):
        for i in range(nx):
            if dz[0, j, i] <= 1e-6: continue
            
            # Surface PAR (Input is NSWF)
            # Ensure non-negative (night time NSWF can be negative)
            i_top = max(0.0, nswr[j, i] * par_fraction)
            
            for k in range(nz):
                dz_c = dz[k, j, i]
                
                if dz_c > 1e-6:
                    # 1. Calculate Attenuation Coefficient for this layer
                    #    More phyto = Darker water
                    k_ext = k_water + (k_chl * psum[k, j, i])
                    
                    # 2. Calculate Light at Bottom of layer (Beer-Lambert Law)
                    #    I_bot = I_top * exp(-k * z)
                    i_bot = i_top * np.exp(-k_ext * dz_c)
                    
                    # 3. Calculate Average Light experienced by biology in this layer
                    #    Integral average: (I_top - I_bot) / (k * z)
                    if k_ext * dz_c > 1e-5:
                        i_avg = (i_top - i_bot) / (k_ext * dz_c)
                    else:
                        # Limit for very thin layers or clear water to avoid div/0
                        i_avg = i_top
                        
                    par_3d[k, j, i] = i_avg
                    
                    # 4. Pass light down to next layer
                    i_top = i_bot
                else:
                    par_3d[k, j, i] = 0.0
                    
    return par_3d


@njit(parallel=True, fastmath=True)
def biology_n2p2z2d2_numba(no3, nh4, ps, pl, zs, zl, ds, dl, temp, par_3d, dz, dt):
    """
    N2P2Z2D2 Ecosystem Model
    Tracer Order: NO3, NH4, PS, PL, ZS, ZL, DS, DL
    """
    nz, ny, nx = no3.shape
    
    # --- PARAMETERS ---
    # 1. Growth & Nutrients
    mu_s     = 0.6   # Max Growth Ps (1/d)
    mu_l     = 1.2   # Max Growth Pl (1/d)
    kn_no3_s = 0.5   # Half-sat NO3 Small
    kn_nh4_s = 0.1   # Half-sat NH4 Small (High affinity!)
    kn_no3_l = 2.0   # Half-sat NO3 Large
    kn_nh4_l = 0.5   # Half-sat NH4 Large
    psi      = 1.5   # Ammonium Inhibition term ((mmol/m3)^-1)
    r_nit    = 0.05  # Nitrification rate (NH4->NO3) (1/d)
    
    # 2. Grazing (Micro-Zoo Zs -> Ps)
    g_max_s  = 0.6
    kg_s     = 0.5
    
    # 3. Grazing (Meso-Zoo Zl -> Pl, Zs, Ds)
    g_max_l  = 0.5
    kg_l     = 1.0
    pref_pl  = 0.7   # Zl prefers Diatoms
    pref_zs  = 0.3   # Zl eats Micro-Zoo
    
    # 4. Mortality & Loss
    m_p      = 0.05  # Phyto linear mort
    m_z_s    = 0.1   # Zs linear mort (Excretion)
    m_z_l    = 0.1   # Zl linear mort (Excretion)
    m_z_quad = 0.05  # Quadratic closure (Predation by higher trophic)
    rem_d    = 0.1   # Remineralization D -> NH4
    
    # 5. Stoichiometry/Routing
    gamma    = 0.7   # Assimilation efficiency (30% is sloppy feeding -> Detritus)
    
    # Time conversion
    dt_day = dt / 86400.0
    
    # Initialize Outputs
    no3_new = np.zeros_like(no3)
    nh4_new = np.zeros_like(nh4)
    ps_new  = np.zeros_like(ps)
    pl_new  = np.zeros_like(pl)
    zs_new  = np.zeros_like(zs)
    zl_new  = np.zeros_like(zl)
    ds_new  = np.zeros_like(ds)
    dl_new  = np.zeros_like(dl)

    for j in prange(ny):
        for i in range(nx):
            if dz[0, j, i] <= 1e-6: continue

            for k in range(nz):
                if dz[k, j, i] <= 1e-6: continue
                
                # Load State
                NO3 = no3[k, j, i]; NH4 = nh4[k, j, i]
                Ps  = ps[k, j, i];  Pl  = pl[k, j, i]
                Zs  = zs[k, j, i];  Zl  = zl[k, j, i]
                Ds  = ds[k, j, i];  Dl  = dl[k, j, i]
                T   = temp[k, j, i]; PAR = par_3d[k, j, i]
                
                # --- A. LIMITING FUNCTIONS ---
                val_T = np.exp(0.063 * T)
                
                # Light Lim (Smith Function)
                # Ps is efficient at low light, Pl needs high light
                f_light_s = PAR / (PAR + 10.0)
                f_light_l = PAR / (PAR + 20.0)
                
                # Nutrient Lim (Wroblewski Model with NH4 Inhibition)
                # NH4 is preferred. NO3 is used only if NH4 is low.
                
                # Small Phyto Nutrients
                lim_nh4_s = NH4 / (kn_nh4_s + NH4)
                lim_no3_s = (NO3 / (kn_no3_s + NO3)) * np.exp(-psi * NH4)
                tot_nut_s = lim_nh4_s + lim_no3_s
                # Scale total to max 1.0 (Approximate)
                scale_s = min(1.0, tot_nut_s)
                
                # Large Phyto Nutrients
                lim_nh4_l = NH4 / (kn_nh4_l + NH4)
                lim_no3_l = (NO3 / (kn_no3_l + NO3)) * np.exp(-psi * NH4)
                tot_nut_l = lim_nh4_l + lim_no3_l
                scale_l = min(1.0, tot_nut_l)
                
                # Fraction of growth supported by NO3 vs NH4
                frac_no3_s = lim_no3_s / (tot_nut_s + 1e-10)
                frac_no3_l = lim_no3_l / (tot_nut_l + 1e-10)
                
                # --- B. GROWTH ---
                # Realized Growth Rate
                r_growth_s = mu_s * val_T * min(f_light_s, scale_s) * Ps
                r_growth_l = mu_l * val_T * min(f_light_l, scale_l) * Pl
                
                # Uptake Fluxes
                uptake_no3_s = r_growth_s * frac_no3_s
                uptake_nh4_s = r_growth_s * (1.0 - frac_no3_s)
                
                uptake_no3_l = r_growth_l * frac_no3_l
                uptake_nh4_l = r_growth_l * (1.0 - frac_no3_l)
                
                # --- C. GRAZING ---
                # 1. Micro-Zoo (Zs) eats Ps
                graz_s_tot = g_max_s * val_T * (Ps / (kg_s + Ps)) * Zs
                
                # 2. Meso-Zoo (Zl) eats Pl AND Zs
                # Preference weighted food
                food_l = (pref_pl * Pl) + (pref_zs * Zs)
                graz_l_tot = g_max_l * val_T * (food_l / (kg_l + food_l)) * Zl
                
                # Split Zl grazing
                graz_l_on_pl = graz_l_tot * (pref_pl * Pl / (food_l + 1e-10))
                graz_l_on_zs = graz_l_tot * (pref_zs * Zs / (food_l + 1e-10))
                
                # --- D. SOURCES / SINKS ---
                
                # 1. Nitrification (NH4 -> NO3)
                # Bacteria convert NH4 to NO3. Inhibited by light (optional, ignored here for simple offline)
                nitrif = r_nit * NH4
                
                # 2. Mortality / Excretion Terms
                # Phyto Mortality -> Detritus (Small -> Small Det, Large -> Large Det)
                mort_ps = m_p * Ps
                mort_pl = m_p * Pl
                
                # Zoo Excretion (Linear) -> NH4 (Metabolic waste)
                exc_zs = m_z_s * Zs
                exc_zl = m_z_l * Zl
                
                # Zoo Mortality (Quad) -> Large Detritus (Dead bodies sink fast)
                mort_zs_quad = m_z_quad * Zs**2
                mort_zl_quad = m_z_quad * Zl**2
                
                # Sloppy Feeding (Unassimilated grazing) -> Detritus
                # 30% of grazing is poop/mess
                sloppy_s = graz_s_tot * (1.0 - gamma)     # Zs sloppy -> Small Det
                sloppy_l = graz_l_tot * (1.0 - gamma)     # Zl sloppy -> Large Det
                
                # Remineralization (Det -> NH4)
                remin_ds = rem_d * Ds
                remin_dl = rem_d * Dl  # Large det might remineralize slower, but assume same for now
                
                # --- E. DERIVATIVES ---
                
                # NO3: -Uptake + Nitrification
                dNO3 = -uptake_no3_s - uptake_no3_l + nitrif
                
                # NH4: -Uptake - Nitrification + Excretion + Remin
                dNH4 = -uptake_nh4_s - uptake_nh4_l - nitrif + \
                       exc_zs + exc_zl + remin_ds + remin_dl
                       
                # Ps: +Growth - Grazing(Zs) - Mort
                dPs = r_growth_s - graz_s_tot - mort_ps
                
                # Pl: +Growth - Grazing(Zl) - Mort
                dPl = r_growth_l - graz_l_on_pl - mort_pl
                
                # Zs: +Assim(Ps) - Grazing(Zl) - Excretion - Mort
                dZs = (graz_s_tot * gamma) - graz_l_on_zs - exc_zs - mort_zs_quad
                
                # Zl: +Assim(Pl, Zs) - Excretion - Mort
                dZl = (graz_l_tot * gamma) - exc_zl - mort_zl_quad
                
                # Ds (Small Det): +Ps_Mort + Zs_Sloppy - Remin
                dDs = mort_ps + sloppy_s - remin_ds
                
                # Dl (Large Det): +Pl_Mort + Zl_Sloppy + Dead_Zoos - Remin
                dDl = mort_pl + sloppy_l + mort_zs_quad + mort_zl_quad - remin_dl
                
                # --- F. UPDATE ---
                no3_new[k, j, i] = NO3 + dNO3 * dt_day
                nh4_new[k, j, i] = NH4 + dNH4 * dt_day
                ps_new[k, j, i]  = Ps  + dPs  * dt_day
                pl_new[k, j, i]  = Pl  + dPl  * dt_day
                zs_new[k, j, i]  = Zs  + dZs  * dt_day
                zl_new[k, j, i]  = Zl  + dZl  * dt_day
                ds_new[k, j, i]  = Ds  + dDs  * dt_day
                dl_new[k, j, i]  = Dl  + dDl  * dt_day
                
    return no3_new, nh4_new, ps_new, pl_new, zs_new, zl_new, ds_new, dl_new


@njit(parallel=True, fastmath=True)
def sinking_dual_numba(ds, dl, dz, dt):
    """
    Sinks two classes of Detritus at different speeds.
    """
    nz, ny, nx = ds.shape
    ds_new = np.zeros_like(ds)
    dl_new = np.zeros_like(dl)
    
    # Sinking Speeds (m/day)
    w_slow = 1.0   # Small Detritus
    w_fast = 20.0  # Large Detritus (Fecal pellets, dead diatoms)
    
    # Convert to m/s
    ws_s = w_slow / 86400.0
    wl_s = w_fast / 86400.0
    
    for j in prange(ny):
        for i in range(nx):
            if dz[0, j, i] <= 1e-6: continue
            
            flux_in_s = 0.0
            flux_in_l = 0.0
            
            for k in range(nz):
                dz_c = dz[k, j, i]
                if dz_c > 1e-6:
                    # Flux Out = Speed * Concentration
                    flux_out_s = ws_s * ds[k, j, i]
                    flux_out_l = wl_s * dl[k, j, i]
                    
                    # Update
                    ds_new[k, j, i] = ds[k, j, i] + dt * (flux_in_s - flux_out_s) / dz_c
                    dl_new[k, j, i] = dl[k, j, i] + dt * (flux_in_l - flux_out_l) / dz_c
                    
                    # Pass flux down
                    flux_in_s = flux_out_s
                    flux_in_l = flux_out_l
                else:
                    ds_new[k, j, i] = 0.0
                    dl_new[k, j, i] = 0.0
                    flux_in_s = 0.0
                    flux_in_l = 0.0
                    
    return ds_new, dl_new