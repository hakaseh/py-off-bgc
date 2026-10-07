# YAML files explained

* `BGC_MODEL`: BGC model (choices: `NEMURO`, `FENNEL06`)
* `DT_SEC`: Time step in seconds
* `USE_GPU`: Use GPU (true) or CPU (false)
* `MAX_THREADS`: Max number of threads for parallelization (optional and only relevant for `USE_GPU: false`)
* `LON_BOUNDS`: Western and eastern limit of the model domain
* `LAT_BOUNDS`: Southern and northern limit of the model domain
* `DEPTH_BOUNDS`: Upper and lower vertical limit of the model domain
* `DATE_BOUNDS`: First and last date of the model simulation

## Supported forcing

GLORYS12V1, GOEPR, ERA5


