# Configuration Guide (`config.yaml`)

This YAML configuration file controls all aspects of the `py-off-bgc` offline biogeochemical simulator, from spatial subsetting during pre-processing to physical time-stepping and forcing data definitions.

## 1. Domain & Time Bounds
These variables define the spatiotemporal domain of the simulation. They are used by both `prep.py` (to subset raw data) and `run.py` (to define the simulation loop).

| Variable | Type | Description |
| :--- | :--- | :--- |
| `LON_BOUNDS` | `[float, float]` | Minimum and maximum longitude (e.g., `[140, 150]`). |
| `LAT_BOUNDS` | `[float, float]` | Minimum and maximum latitude (e.g., `[40, 50]`). |
| `DEPTH_BOUNDS` | `[float, float]` | Depth range in meters. Use `null` for the surface or bottom (e.g., `[null, 500]`). |
| `DATE_BOUNDS` | `[str, str]` | Start and end dates in `YYYYMMDD` format (e.g., `["20240101", "20241231"]`). |

## 2. Pre-processing (`prep.py`)
Variables used exclusively to prepare boundary conditions and climatology.

| Variable | Type | Description |
| :--- | :--- | :--- |
| `OCEAN_SOURCE` | `str` | Identifier for the ocean forcing dataset (e.g., `"copernicus_goepr"`). |
| `ATMOS_SOURCE` | `str` | Identifier for the atmospheric forcing dataset (e.g., `"cds_era5"`). |
| `REF_OCEAN` | `str` | Filepath to a reference ocean netCDF file used to extract the land/ocean mask and grid geometry. |
| `VAR_OCEAN` | `str` | The variable name inside `REF_OCEAN` used to generate the mask (e.g., `"thetao_mean"`). |
| `GLODAP_DIR` | `str` | Directory containing the GLODAPv2 mapped climatology files for restoring/initialization. |

## 3. Simulation Parameters (`run.py`)
Core parameters for the offline Taichi-accelerated physics and biology.

| Variable | Type | Description |
| :--- | :--- | :--- |
| `BGC_MODEL` | `str` | The biogeochemical model to run (e.g., `"NEMURO"`, `"FENNEL06"`, `"NO_SMS"`). |
| `DT_SEC` | `float` | Physics time-step in seconds. Must satisfy the 3D CFL stability condition (e.g., `3600.0`). |
| `USE_GPU` | `bool` | Set to `true` to force Taichi to use the GPU, or `false` for CPU fallback. |
| `MAX_THREADS` | `int` | Maximum threads for CPU execution. Use `null` for auto-detect. |

## 4. Forcing Data (`FORCING`)
This dictionary maps physical oceanographic and atmospheric variables to their respective netCDF files.

For each variable (`T`, `S`, `U`, `V`, `SW`, `WIND`, `WIND_U`, `WIND_V`, `ICE`, `K`), you must define:
*   `FILE`: Path to the netCDF file (supports wildcards like `*.nc`). Set to `null` to disable/use defaults.
*   `NAME`: The exact variable name inside the netCDF file.
*   `SAVE`: (Optional) Boolean `true`/`false`. If true, this forcing field will be written to the daily output files.

**Wind Configuration Note:** 
You must provide *either* wind magnitude (`WIND`) *or* the zonal/meridional components (`WIND_U` and `WIND_V`). Do not define both. If neither is defined, wind stress defaults to zero.

## 5. Restoring & Sponge Layer
Controls lateral and bottom nudging toward climatology to prevent model drift and handle open boundary conditions.

| Variable | Type | Description |
| :--- | :--- | :--- |
| `SPONGE_WIDTH` | `int` | Number of grid cells from the edge to apply lateral sponge restoring. |
| `TAU_LATERAL` | `float` | Relaxation timescale (in seconds) for the lateral sponge boundaries. |
| `TAU_BOTTOM` | `float` | Relaxation timescale (in seconds) for the ocean bottom. |
| `TAU_COAST` | `float` | Relaxation timescale (in seconds) for coastal land/ocean boundaries. |
| `IS_GLOBAL` | `bool` | Set to `true` if the domain wraps around the Earth (enables periodic X-boundaries). |

## 6. Initialization & State
Defines the initial states and target states for the simulator.

| Variable | Type | Description |
| :--- | :--- | :--- |
| `RESTART_FILE` | `str` | Path to a previous model output `.nc` file to use as the initial condition. Set to `null` for cold start. |
| `CLIM_FILE` | `str` | Path to the 3D biogeochemical climatology file used for nudging and restoring. Set to `null` if no restoring is desired. |