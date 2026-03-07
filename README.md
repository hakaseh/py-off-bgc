# py-off-bgc: a Python-based offline ocean biogeochemistry simulator


# Supported ocean models

- JCOPE NCEP CFS
- LORA 2016-2023 JRA55do
- BRAN2020 JRA55do

## BRAN2020
- T and S have different lon-lat grid from U and V. It must be interpolated during the simulation.
- Dimension names need renaming.
- Required time step: 600 s due to the higher latitude requiring finer time step.
- With 60S-60N and full depth, it takes 1.5 hours to simulate one day.

## LORA
- Required time step: 1200 s
- 

# Supported BGC models

- NPZD based on Sasai et al. (2022)
- NEMURO based on Kishi et al. (2007)
- eNEMURO based on Yoshie et al. (2011)

# Simulation time (per day)

- BRAN2020-NPZD, 60S-60N, full depth, 600 s, 90 min
- LORA-NEMURO, full domain, 1000 m, 1200 s, 5 min
- LORA-NPZD, full domain, 1000 m, 1200 s, 1 min
- OFES