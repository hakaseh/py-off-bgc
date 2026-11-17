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

To our knowledge, the Operational Mercator Ocean biogeochemical global ocean analysis and forecast system at 1/4 degree[^1] is the only existing model that provides both the reanalysis and forecast of global ocean biogeochemistry. The system updates the reanalysis and 10-day prediction daily. Beyond this lead time, the SINTEX-F2bio system provides seasonal prediction of sea surface chlorophyll-a and nitrate up to 7 months in advance updated monthly based on a 10-member ensemble [@doi_seasonal_2026]. Additionally, there are several reanalysis and sub-seasonal forecasting systems developed for specific regions [@fennel_advancing_2019].

**Py-off-bgc** contributes to these reanalysis and forecast products of ocean biogeochemistry by running ocean biogeochemical models using prescribed ocean dynamics. To our knowledge, no such general offline framework exits except for the transport matrix method [@khatiwala_computational_2007], which requires a deep understanding of advanced linear algebra and computational mathematics to implement and use. In contract, **py-off-bgc** is designed to be as generic as possible, allowing offline simulation of ocean biogeochemistry using any ocean reanalysis or forecast product as long as the product provides daily-mean three-diemensional fields of seawater temperature, salinity, and horizontal velocity in netCDF format.


[^1]: Global Ocean Biogeochemistry Analysis and Forecast. E.U. Copernicus Marine Service Information (CMEMS). Marine Data Store (MDS). DOI: 10.48670/moi-00015 (Accessed on 17 NOV 2026)

# Software design

## Code structure

Nearly all numerical ocean models are written in Fortran, even though interpreted programming languages such as Python are more intuitive and becoming computationally competitive with compiled languages [@veros]. This tradition can be a bottleneck in getting started with ocean modelling particularly for those who were trained in Python or similar modern languages. **Py-off-bgc** is written solely in Python to make it accessible to students and early-career researchers of oceanography who are more familiar with Python than Fortran. A major advantage of Python over Fortran is the availability of software libraries that can be easily adapted and updated. In particular, **py-off-bgc** uses the following Python libraries for making it modern, flexible, and adaptable:

* **Taichi** for acceleration of computing to be compatible with Fortran and enabling GPU computing. It also supports automatic differentiation, which can be used for parameter tuning and data assimilation.
* **Xarray** for facilitating I/O of daily and three-dimensional ocean data.
* **YAML** for readability of configuration files.


## Input forcing data

For simulation of tracer transport (see below), requires temperature, salinity, and horizontal velocity. For simulation of photosynthetic growth of phytoplankton, shortwave radiation. For regions covered by sea ice, sea-ice concentration. wind speed for gases such as dissolved oxygen and carbon dioxide. 

## Advection and diffusion schemes

Tracer advection is solved using a Forward-Time Upwind-Space (FTUS) scheme, modified with a hard-capping constraint to strictly preserve positivity and zero-gradient Neumann conditions at land boundaries. Vertical mixing is handled via a simplified K-Profile Parameterization (KPP) in the surface mixed layer [@large_oceanic_1994], transitioning to a stratification-dependent internal wave mixing scheme in the ocean interior [@gargett_vertical_1984]. The resulting vertical diffusion is integrated using an explicit Forward-Time Centered-Space (FTCS) finite volume scheme, dynamically constrained by a Von Neumann stability limiter to ensure numerical robustness across varying grid resolutions.

## Initial and boundary conditions and restoring



# A pilot use case

To demonstrate the simulation capability of **py-off-bgc**, we show the results of using NEMURO [@kishi_nemurolower_2007].



# Research impact statement



# AI usage disclosure

Gemini Pro was used for enhancing code writing, while we verified its accuracy through manual review and sometimes by conducting sensitivity experiments.

# Acknowledgements

We thank Hidehiro Fujio for computational assistance and Ruochao Zhang for the operation of JCOPE2M. This work was supported by JSPS KAKENHI Grant Number JP24H02226.

# References


