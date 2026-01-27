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



### 2. Experiments

#### 2.1 Oracle
Run the following:
```
python scripts/run_realdata_strategies.py   taxi_yellow_green_rideshare_distinct_march_to_apr2020_triplets.parquet --modes homo,binwise,hetero --periodicity 168 --n_components 10 --init_cycles 3 --conf_alpha 0.10 --conf_buf 100000 --zero_samples 3000 --zero_top_frac 0.6 --zero_pool_mult 10 --out_root out_real/all3cv1oracle --bin_edges 0,1,2,3,5,8,13,21,34,55,89
```
Use output directories `--out_root out_real/all3cv<1,2,3,4,5>oracle` to obtain the five Oracle evaluations in the paper.


#### 2.2 Non-oracle (Ex-ante)

##### 2.2.1 Homoskedastic Conformal
Example command for the homoskedastic results varying the threshold $\tau$: 
```
python scripts/run_realdata_strategies.py taxi_yellow_green_rideshare_distinct_march_to_apr2020_triplets.parquet  --modes homo  --periodicity 168  --n_components 10  --init_cycles 3  --conf_alpha 0.10  --conf_buf 100000  --zero_samples 3000  --zero_top_frac 0.6  --zero_pool_mult 10  --out_root out_real/homo_tau05/homo1tau05cv1  --exante_tau 0.5
```
Use output directories `--out_root out_real/homo_tau05/homo1tau05cv<1,2,3,4,5>` to obtain the five non-oracle evaluations in the paper. Vary the "tau" portion of the directory, along with the `--exante_tau 0.5` argument


##### 2.2.2 Binwise Conformal

Command for the binwise results: 
```
python scripts/run_realdata_strategies.py taxi_yellow_green_rideshare_distinct_march_to_apr2020_triplets.parquet  --modes binwise  --periodicity 168  --n_components 10  --init_cycles 3  --conf_alpha 0.10  --conf_buf 100000  --zero_samples 3000  --zero_top_frac 0.6  --zero_pool_mult 10  --out_root out_real/binwise_nonoracle/binwise_nonoracle_cv1  --bin_edges 0,1,2,3,5,8,13,21,34,55,89 
```
Use output directories `--out_root out_real/binwise_nonoracle/binwise_nonoracle_cv<1,2,3,4,5>` to obtain the five non-oracle evaluations in the paper.

##### 2.2.3 Heteroskedastic Conformal

Command for the heteroskedastic results: 
```
python scripts/run_realdata_strategies.py taxi_yellow_green_rideshare_distinct_march_to_apr2020_triplets.parquet  --modes hetero  --periodicity 168  --n_components 10  --init_cycles 3  --conf_alpha 0.10  --conf_buf 100000  --zero_samples 3000  --zero_top_frac 0.6  --zero_pool_mult 10  --out_root out_real/hetero_nonoracle/hetero_nonoracle_cv1
```
Use output directories `--out_root out_real/hetero_nonoracle/hetero_nonoracle_cv<1,2,3,4,5>` to obtain the five non-oracle evaluations in the paper.
