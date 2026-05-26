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
source activate env_py-off-bgc
```

Via Pip
```
python -m venv env_py-off-bgc
source env/bin/activate
pip install -r requirements.txt
```

## An example: GOEPR_ERA5_Hokkaido

* Download the forcing dataset from [Zenodo](https://zenodo.org/records/19703262).
* Rename the above dataset directory as **GOEPR_ERA5_Hokkaido** and place it under `py-off-bgc/input`.
* Create the BGC climatology.
* In your root directory, run the simulation
```
cd;
cd py-off-bgc
python -m run.run_goepr
```
## Supported ocean models

* JCOPE NCEP CFS
* LORA 2016-2023 JRA55do
* BRAN2020 JRA55do
* OFES2_NP10 JRA55do

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
- NEMURO based on Kishi et al. (2007)

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


# Simulation time (per day)

- BRAN2020-NPZD, 60S-60N, full depth, 600 s, 90 min
- LORA-NEMURO, full domain, 1000 m, 1200 s, 5 min
- LORA-NPZD, full domain, 1000 m, 1200 s, 1 min
- OFES2_NP10, JCOPE-T domain, upper 300m, 600s, 