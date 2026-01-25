# uncertainty_ssmf

Under construction!!!


# Distributed SSMF
This repository houses a [Distributed Shifting Seasonal Matrix Factorization](https://ieeexplore.ieee.org/document/11196218) model presented at IEEE HPEC 2025. 

Joint work with [@breecummins](https://github.com/breecummins).

Paper abstract:

> Forecasting methods ingest time series data and provide a prediction of future events. One sophisticated forecasting algorithm is called the Shifting Seasonal Matrix Factorization (SSMF) method, which is notable for its ability to model repeating patterns in data as well as abrupt global changes in system behavior (regime shifts). SSMF as originally implemented does not scale well with input data size. To address this issue, we provide an MPI parallelization scheme with 2D matrix partitioning, sparsity-awareness, and better regime lifecycle management. We demonstrate the improved scaling and performance using NYC taxicab and rideshare data.


## To install:
```
conda env create -f environment.yml
```

## Activate environment & dependencies:
```
source ./activate_distributed_env.sh
```
(see the referenced file for specific module dependencies)

## Download & Clean Data:

Source data: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

I used these files:

- yellow_tripdata_2020-03.parquet
- yellow_tripdata_2020-04.parquet
- green_tripdata_2020-03.parquet
- green_tripdata_2020-04.parquet
- fhv_tripdata_2020-03.parquet
- fhv_tripdata_2020-04.parquet

Clean data with notebook:
```
nytaxi_processing.ipynb
```

### Create Scaled Dataset (optional)

This is required for scaling experiments, but can be skipped when just getting the model up and running on the above dataset(s) produced from the notebook.

Use the script:
```
scale_data.py
```

Example run to create a new synthetic dataset with 10x matrix dimensions:
```
python scale_dataset.py taxi_yellow_green_rideshare_march_to_apr2020_triplets.parquet scaled_data10x.parquet --scaling-factor 10
```

## Models

### 1. Baseline SSMF model:

See:
  - Repo: https://github.com/kokikwbt/ssmf/tree/main
  - Paper: https://proceedings.neurips.cc/paper/2021/hash/1fb2a1c37b18aa4611c3949d6148d0f8-Abstract.html

Our slight alteration designed to accumulate forecasts at each step:
```
ssmf_forecast.py
```
This uses `ncp.py` from the original SSMF repository. We include a slightly (non-substantively) modified version in this repository.

### 2. SSMF Tuples

This is our version of SSMF that accepts a tuple stream of input. Can be found in:
```
ssmf_tuples.py
```
You can run via:
```
python ssmf_tuples.py taxi_yellow_green_rideshare_distinct_march_to_apr2020_triplets.parquet
```

### 3. Distributed SSMF

Our Distributed SSMF model is found in:
```
ssmf_mpi_2d_dist_init.py
```
This includes the Distributed NCP initialization model `ncp_distributed_2d()` , which is in the above script. Supporting functions are found in:
```
ncp_distributed.py
```
