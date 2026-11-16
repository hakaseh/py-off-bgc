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
    orcid: 
    affiliation: 1
affiliations:
  - name: Japan Agency for Marine-Earth Science and Technology (JAMSTEC), Yokohama, Japan
    index: 1

date: 1 January 2027
bibliography: paper.bib
---

# Summary

**Py-off-bgc** is a general framework for conducting three-dimensional numerical simulation of ocean biogeochemistry using prescribed seawater temperature, salinity, and horizontal velocity fields that can be derived from ocean reanalysis and forecast products. It expands the capability of operational oceanography beyond ocean physics without additional legacy Fortran coding. Written entirely in Python and using state-of-the-art libraries, **py-off-bgc** is also made practical for students and early-career ocean professionals [@li_early_2025] interested in ocean biogeochemical modelling. Hence, **py-off-bgc** contributes to a *predicted* and *accessible* ocean we want by 2030 described by [the United Nations Decade of Ocean Science for Sustainable Development](https://oceandecade.org).

# Statement of need

## Offline simulation

An offline simulation in ocean modelling refers to a situation in which tracer transport is calculated using the output of ocean dynamics, as opposed to being simulated simultaneously with ocean physics. Offline simulations are computationally cheaper than online simulations, and therefore are often used for simulations that require a long integration (e.g., spin-up) or many realizations (e.g., ensembles).

## Ocean reanalysis and forecast products

Ocean reanalysis refers to best estimates of ocean variables such as temperature, salinity, and velocity, that are reproduced by combining observations with a numerical model known as data assimilation. This technique is used to generate realistic initial conditions for conducting forecasts on operational basis.

With increasing computational resources and observing systems, various institutions around the world make ocean reanalysis and forecast products available today [@balmaseda_ocean_2015]. However, nearly all these products do not include biogeochemistry, limiting our understanding and capability to ocean physics. Adding a biogeochemical model into an existing and established reanalysis and forecast system (online coupling) requires  To address this issue, we develop **py-off-bgc** that can conduct offline biogeochemical simulations driven by those ocean reanalysis products. 

## Python

Nearly all numerical ocean models are written in Fortran, even though interpreted programming languages such as Python are more intuitive and becoming computationally competitive with compiled languages [@veros]. This tradition can be a bottleneck in getting started with ocean modelling particularly for those who were trained in Python or similar modern languages. **Py-off-bgc** is written solely in Python to make it accessible to students and early-career researchers of oceanography who are more familiar with Python than Fortran. A major advantage of Python over Fortran is the availability of software libraries that can be easily adapted and updated. In particular, **py-off-bgc** uses the following Python libraries for making it modern, flexible, and adaptable:

* **Taichi** for acceleration of computing to be compatible with Fortran and enabling GPU computing. It also supports automatic differentiation, which can be used for parameter tuning and data assimilation.
* **Xarray** for facilitating I/O of daily and three-dimensional ocean data.
* **YAML** for readability of configuration files.



# State of the field

## Generic offline framework

Existing numerical ocean models allow offline biogeochemical simulation, but they are designed only to work with their own configurations. In contract, **py-off-bgc** is designed to be as generic as possible, allowing offline simulation of ocean biogeochemistry using any ocean reanalysis or forecast product as long as the product provides daily-mean three-diemensional fields of seawater temperature, salinity, and horizontal velocity in netCDF format. This generic 


# Software design

## Code structure

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


