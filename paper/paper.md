---
title: 'py-off-bgc: A Python-based offline simulator for ocean biogeochemistry'
tags: 
  - Numerical modelling
  - Python
  - Oceanography
  - Marine biogeochemistry

authors:
  - name: Hakase Hayashida
    orcid: 0000-0002-6349-4947
    affiliation: 1
    corresponding: true
  - name: Sudeep Das
    orcid: 0000-0001-8207-8351
    affiliation: 1
affiliations:
  - name: Japan Agency for Marine-Earth Science and Technology (JAMSTEC), Yokohama, Japan
    index: 1

date: 1 January 2027
bibliography: paper.bib
---

# Summary

**Py-off-bgc** is a general framework for conducting three-dimensional numerical simulation of ocean biogeochemistry using prescribed seawater temperature, salinity, and horizontal velocity fields that can be derived from ocean reanalysis and forecast products. This offline approach allows independent biogeochemical model development without legacy Fortran coding, while permitting flexible coupling to any combination of forcing data and a biogeochemical model. Written entirely in Python and using state-of-the-art libraries, **py-off-bgc** is also made practical for students and early-career ocean professionals [@li_early_2025] interested in ocean biogeochemical modelling. As such, **py-off-bgc** contributes to both a *predicted* ocean and an *accessible* ocean we want by 2030 described by [the United Nations Decade of Ocean Science for Sustainable Development](https://oceandecade.org).

# Statement of need

Ocean reanalysis refers to state estimates of ocean variables such as temperature and salinity by integrating observations into numerical models. With increasing computational resources and observing systems, various institutions around the world generate ocean reanalysis products that are used for understanding the historical and present ocean climate, as well as for initializing seasonal-to-decadal forecasts [@balmaseda_ocean_2015]. However, nearly all these products are limited to physical variables, therefore our understanding on the biogeochemical properties of the ocean lags behind and their predicability remain largely unxplored despite their significance in the Earth system.

Here we introduce **py-off-bgc**, a general framework that allows to run an ocean biogeochemical model *offline*, which means that the transport of biogeochemical tracers is calculated using seawater temperature, salinity, and horizontal velocity derived from a reanalysis or forecast product. This offline approach is flexible, as users can choose any combination of the input data (ocean physics) and the output model (ocean biogeochemistry), providing opportunities for potential use cases of reanalysis and forecast products in biogeochemical modelling. Hence, **py-off-bgc** aims to fill the major gap in operational capability of ocean state estimate and prediction.

Another remarkable feature of **py-off-bgc** is that it is written entirely in Python, instead of Fortran as in nearly all existing numerical ocean models. Many students and early-career ocean professionals are more familiar with Python and other similar interpreted languages than compiled languages such as Fortran. Therefore, **py-off-bgc** may be a suitable software for getting started with ocean biogeochemical modelling and addressing research questions that require numerical experiments.


# State of the field

Numerical ocean models typically consist of an offline configuration for biogeochemistry, but it is designed to work only internally. To our knowledge, the only general offline framework built to work with any model is the transport matrix method [e.g., @khatiwala_computational_2007], but it requires a deep understanding of advanced linear algebra and computational mathematics to implement and use. In contrast, **py-off-bgc** is designed to be practical, allowing offline biogeochemical simulation simply by providing daily-mean 3D fields of seawater temperature, salinity, and horizontal velocity in netCDF format.

To our knowledge, the Operational Mercator Ocean biogeochemical global ocean analysis and forecast system at 1/4 degree[^1] is the only existing model that provides both the reanalysis and forecast of global ocean biogeochemistry. The system updates the reanalysis and 10-day prediction daily. Beyond this lead time, the SINTEX-F2bio system recently initiated an operational seasonal prediction of global sea surface chlorophyll-a and nitrate up to 7 months in advance which is updated monthly based on a 10-member ensemble [@doi_seasonal_2026]. Additionally, there are several regional systems providing reanalysis and sub-seasonal forecasts of ocean biogeochemistry [@fennel_advancing_2019]. **Py-off-bgc** aims to expand this collection to contribute to advancing the operational capability of ocean biogeochemical modelling.

[^1]: Global Ocean Biogeochemistry Analysis and Forecast. E.U. Copernicus Marine Service Information (CMEMS). Marine Data Store (MDS). DOI: 10.48670/moi-00015 (Accessed on 17 NOV 2026)


# Software design

## Code structure

Nearly all numerical ocean models are written in Fortran, even though interpreted programming languages such as Python are more intuitive and becoming computationally competitive with compiled languages [@veros]. This tradition can be a bottleneck in getting started with ocean modelling particularly for those who are trained in Python or similar modern languages. **Py-off-bgc** is written solely in Python to make it accessible to students and ECOPs who are more familiar with Python than Fortran. A major advantage of Python is the availability of software libraries that can be easily adapted and updated. In particular, **py-off-bgc** uses the following Python libraries for making it flexible and adaptable:

* `Taichi` for acceleration of computing to be compatible with Fortran and enabling GPU computing. It also supports automatic differentiation, which can be used for parameter tuning and data assimilation.
* `Xarray` for facilitating I/O of daily 3D ocean data.
* `YAML` for readability of configuration files.

One obvious trade-off of **py-off-bgc** is the lack of support for ocean tiling, which is a computational method to divide the spatial domain into sub-domains so they can be processed simultaneously using multiple CPU cores or computer nodes. It is commonly used in large-scale and/or high-resolution ocean modelling that requires significant computational power.

## Input forcing data

For simulation of tracer transport (see below), requires temperature, salinity, and horizontal velocity. For simulation of photosynthetic growth of phytoplankton, shortwave radiation. For regions covered by sea ice, sea-ice concentration. wind speed for gases such as dissolved oxygen and carbon dioxide. 

## Advection and diffusion schemes

The tracer transport in py-off-bgc is driven by a 3D Total Variation Diminishing (TVD) advection scheme designed to maintain both mass conservation and gradient sharpness. To balance numerical stability with high-order accuracy, the solver employs a hybrid directional approach: a Van Leer limiter is applied horizontally to preserve mesoscale fronts and eddies, while a strictly diffusive MinMod limiter is applied vertically to suppress node-to-node numerical dispersion in thin layers. Furthermore, the flux-form advection incorporates a local volume divergence correction (Advective Form) to prevent artificial tracer accumulation when water compresses against solid boundaries. Because the model directly ingests collocated Arakawa A-grid reanalysis velocities, it utilizes a mask-aware spatial filter to eliminate grid-scale divergence artifacts prior to diagnosing vertical velocity via continuity, enforcing strict no-flux coastal boundaries and a rigid-lid sea floor.

Vertical mixing is handled by an unconditionally stable implicit diffusion solver using the Thomas algorithm, executing immediately after the biological and air-sea flux updates to seamlessly homogenize surface layers. The physical mixing strength is governed by a dynamically parameterized vertical eddy diffusivity ($K_z$) array. Within the prescribed Mixed Layer Depth (MLD), $K_z$ is shaped by a K-Profile Parameterization (KPP)-style polynomial function ($6.75 \sigma (1-\sigma)^2$) to simulate intense wintertime convective overturning [@large_oceanic_1994]. Below the mixed layer, the model transitions to a Brunt-Väisälä frequency-dependent ($N^2$) stratification profile [@gargett_vertical_1984], ensuring physically realistic vertical blending across both open ocean and complex coastal topologies.


## Initial and boundary conditions and restoring



# A pilot use case

To demonstrate the simulation capability of **py-off-bgc**, we show the results of using NEMURO [@kishi_nemurolower_2007].



# Research impact statement



# AI usage disclosure

Gemini Pro was used for enhancing code writing, while we verified its accuracy through manual review and sometimes by conducting sensitivity experiments.

# Acknowledgements

We thank Hidehiro Fujio for computational assistance and Ruochao Zhang for the operation of JCOPE2M. This work was supported by JSPS KAKENHI Grant Number JP24H02226.

# References


