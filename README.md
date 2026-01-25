# Uncertainty SSMF
This repository houses a *Uncertainty-Aware Forecasting with Shifting Seasonal Matrix Factorization* effort awaiting publication. 

Joint work with [@breecummins](https://github.com/breecummins).

Paper abstract:

> Uncertainty estimation remains a difficult problem in machine learning. Where possible, principled uncertainty estimation often requires extensive model alterations, repeated model runs, or expensive sampling-based methods. In this paper, we extend the existing Shifting Seasonal Matrix Factorization (SSMF) model to include meaningful uncertainty estimates without major model modifications. We use a conformal prediction framework with multiple variants to estimate prediction intervals based on a user-specified quantile. Our results show that permitting time-dependent variance in model residuals accurately models coverage over tight prediction intervals.


## To install:
```
conda env create -f environment.yml
```

## Activate environment & dependencies:
```
source ./activate_uncertainty_env.sh
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

Clean data with notebook (found in another related repo - https://github.com/jacobmunson/distributed_ssmf/tree/main):
```
nytaxi_processing.ipynb
```



## Models

### 1. Baseline SSMF 

This is the tuple version of SSMF that accepts a tuple stream of input. Can be found in:
```
ssmf_tuples.py
```
You can run via:
```
python ssmf_tuples.py taxi_yellow_green_rideshare_distinct_march_to_apr2020_triplets.parquet
```


