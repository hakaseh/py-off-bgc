# py-off-bgc: a Python-based offline simulator for ocean biogeochemistry

## 🚀 Getting Started

* Install py-off-bgc

```
git clone https://github.com/hakaseh/py-off-bgc.git
cd py-off-bgc
```

* Create a Python environment for py-off-bgc using either Conda or Pip.

Via Conda
```
conda env create -f environment.yml
source activate env
```

Via Pip
```
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

### An example test case: GOEPR_ERA5_Hokkaido

* Download the forcing dataset from [Zenodo](https://zenodo.org/records/19703262).
* Rename the above dataset directory as **GOEPR_ERA5_Hokkaido** and place it under `py-off-bgc/input`.
* Create the BGC climatology.
* In your root directory, run the simulation
```
cd;
cd py-off-bgc
python3 -m run.run_goepr
```

## Run files (to be moved to run/README.md)
* Rename to required dimension names (lon, lat, depth, time)
* Interpolate spatially and/or temporally (see examples from BRAN2020 for spatial interpolation and GOEPR for temporal interpolation)

## Supported ocean models

* JCOPE-FGO NCEP-CFS
* LORA-NWP 2016-2023 JRA55do
* BRAN2020 JRA55do
* OFES2-NP10 JRA55do

### BRAN2020
- T and S have different lon-lat grid from U and V. It must be interpolated during the simulation.
- Dimension names need renaming.
- Required time step: 600 s due to the higher latitude requiring finer time step.
- With 60S-60N and full depth, it takes 1.5 hours to simulate one day.

### LORA
- Required time step: 1200 s
- 

## Supported BGC models

- NPZD based on Sasai et al. (2022)
- N1P1 based on Fennel et al. (2006)
- FlexPFT based on Kerimoglu et al. (2023)
- NEMURO based on Kishi et al. (2007)

### Adding a new BGC model

* In `source/bgc_models/__init__.py`, add `from .nemuro import Model_NEMURO` and `"NEMURO": Model_NEMURO,`.
* In `source/bgc_models`, add `nemuro.py`.


### Modify the BGC parameters
By default, the model will run with the default set of parameter values declared in the source code. If you want to change these parameter values in your run, add the following to your **run.py** file:

1. Load the parameter set `Params_BGC` of the BGC model of your choice and create a copy `custom_bgc`:

```
from source.bgc_models.npzd_sasai import Params_BGC
custom_bgc = Params_BGC()
```

1. Assign your modified parameter values:

```
custom_bgc.p_res = 0 
custom_bgc.p_exc = 0     
custom_bgc.p_gro = 1
custom_bgc.p_gef = 0.7
custom_bgc.p_zmo = 0.24
custom_bgc.p_pmo = 0.12
```
1. Add `custom_bgc` to the argument for `OfflineSimulator`:

```
sim = OfflineSimulator(
    bgc_model_choice=bgc_model_choice, exp_name=exp_name,
    dt_phys=dt_in_sec, 
    bgc_params=custom_bgc, # <-- HERE!
    mld_threshold=mld_choice, sponge_width=sponge_choice,
    tau_lateral=tau_lateral_choice, tau_bottom=tau_bottom_choice, is_global=is_global_choice
)
```


# Simulation time, CPU and RAM usage
Some computational stats can be calculated using either `/usr/bin/time` (for Linux) or `/opt/homebrew/bin/gtime` (for Mac). For example:

```
/usr/bin/time -v python3 -m run.run_jcope-fgo
/opt/homebrew/bin/gtime -v python3 -m run.run_jcope-fgo
```
Benchmarking with Oyashio (32 CPUs)
* JCOPE-FGO-NEMURO: 1 year, T-domain, 1800s, 3 hours, 6 CPUs, 2.6 GB --> blow up.
* LORA-NWP-NEMURO: 1 year, T-domain, 1800s, 3 hours, 6 CPUs, 2.9 GB --> blow up.
* JCOPE-FGO-NEMURO: 1 year, T-domain, 1200s, 8.5 hours, 685 %, 2.6 GB --> seems stable
* LORA-NWP-NEMURO: 1 year, T-domain, 600s, 15.5 hours, 639 %, 3392064 KB --> unstable.
* LORA-NWP-NEMURO: 1 year, T-domain, 450s, 20.25 hours, 675 %, 3373544 KB --> unstable.
* LORA-NWP-NEMURO: 1 year, T-domain, 450s, 20.25 hours, 675 %, 3373544 KB --> 


Computing time per day of simulation: (To be deleted)
- BRAN2020-NPZD, 60S-60N, full depth, 600 s, 90 min
- LORA-NEMURO, full domain, 1000 m, 1200 s, 5 min
- LORA-NPZD, full domain, 1000 m, 1200 s, 1 min
- OFES2_NP10, JCOPE-T domain, upper 300m, 600s, 
