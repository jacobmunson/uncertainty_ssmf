#!/usr/bin/env python3
from __future__ import annotations
import argparse, time
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ssmf_tuples as mod
from uncertainty.conformal import make_strategy

"""
python scripts/run_realdata_strategies.py \
  taxi_yellow_green_rideshare_distinct_march_to_apr2020_triplets.parquet \
  --modes homo,binwise,hetero \
  --periodicity 24 --n_components 10 --init_cycles 3 \
  --conf_alpha 0.10 --conf_buf 100000 \
  --zero_samples 3000 --zero_top_frac 0.6 --zero_pool_mult 10 \
  --out_root out_real \
  --bin_edges 0,1,2,3,5,8,13,21,34,55,89

python scripts/run_realdata_strategies.py \
    taxi_yellow_green_rideshare_distinct_march_to_apr2020_triplets.parquet \
    --modes homo,binwise,hetero,bestn,inject,kmodels,mc_dropout,kruns \
    --out_root out_real   --periodicity 24   --n_components 10   --conf_alpha 0.10

python scripts/run_realdata_strategies.py \
    taxi_yellow_green_rideshare_distinct_march_to_apr2020_triplets.parquet \
    --modes hetero \
    --out_root out_real   --periodicity 24   --n_components 10   --conf_alpha 0.10

"""

def _auto_bin_edges_from_counts(counts: np.ndarray, num_bins: int = 10) -> np.ndarray:
    """Quantile-based edges from raw trip_count distribution (robust, no model change)."""
    qs = np.linspace(0.0, 0.999, num_bins + 1)
    edges = np.quantile(counts, qs).astype(float)
    # ensure strictly increasing (nudge ties upward a tiny bit)
    for i in range(1, edges.size):
        if edges[i] <= edges[i-1]:
            edges[i] = np.nextafter(edges[i-1], np.inf)
    # force first edge to 0
    edges[0] = 0.0
    return edges

def build_triplet_dict(df: pd.DataFrame):
    triplet_dict = defaultdict(list)
    for r in df.itertuples(index=False):
        triplet_dict[int(r.t_idx)].append((int(r.PU_idx), int(r.DO_idx), float(r.trip_count)))
    T  = int(df.t_idx.max()) + 1
    d1 = int(df.PU_idx.max()) + 1
    d2 = int(df.DO_idx.max()) + 1
    return triplet_dict, T, (d1, d2)

def run_one_mode(args, mode: str, df: pd.DataFrame, edges: np.ndarray | None):
    out_dir = Path(args.out_root) / mode
    out_dir.mkdir(parents=True, exist_ok=True)

    triplet_dict, T, dshape = build_triplet_dict(df)

    model = mod.SSMF(
        triplet_dict=triplet_dict,
        periodicity=args.periodicity,
        n_components=args.n_components,
        max_regimes=args.max_regimes,
        max_iter=args.max_iter,
        alpha=args.learning_rate,
        beta=args.penalty,
        update_freq=args.update_freq,
        float_cost=args.float_cost,
        init_cycles=args.init_cycles,
    )
    model.output_dir = str(out_dir)

    # Attach uncertainty strategy
    strategy_kwargs = dict(
        zero_samples_per_slice=args.zero_samples,
        zero_top_frac=args.zero_top_frac,
        zero_pool_mult=args.zero_pool_mult,
        nz_samples_per_slice=args.nz_samples, 
    )

    strategy_kwargs["exante_tau"] = args.exante_tau


    if mode == "binwise":
        strategy_kwargs["bin_edges"] = edges
        strategy_kwargs["bin_Nmin_zero"] = args.bin_Nmin_zero
    
    elif mode == "bestn":
        strategy_kwargs["n_best"] = args.n_best
        strategy_kwargs["ens_scale_nz"] = args.ens_scale_nz
        strategy_kwargs["ens_scale_z"] = args.ens_scale_z

        if args.diversity_weight is not None:
            strategy_kwargs["diversity_weight"] = args.diversity_weight
        if args.norm_clip is not None:
            strategy_kwargs["norm_clip"] = args.norm_clip
        if args.q_cap is not None:
            strategy_kwargs["q_cap"] = args.q_cap

    elif mode == "kmodels": # This is for KModelEnsembleStrategy
        if args.kmodels_ens_scale_nz is not None:
            strategy_kwargs["ens_scale_nz"] = args.kmodels_ens_scale_nz
        if args.kmodels_ens_scale_z is not None:
            strategy_kwargs["ens_scale_z"] = args.kmodels_ens_scale_z
        if args.kmodels_std_floor is not None:
            strategy_kwargs["std_floor"] = args.kmodels_std_floor
        if args.kmodels_std_rel_floor is not None:
            strategy_kwargs["std_rel_floor"] = args.kmodels_std_rel_floor
        if args.kmodels_jitter_u is not None:
            strategy_kwargs["jitter_u"] = args.kmodels_jitter_u
        if args.kmodels_jitter_v is not None:
            strategy_kwargs["jitter_v"] = args.kmodels_jitter_v
        if args.kmodels_jitter_w is not None:
            strategy_kwargs["jitter_w"] = args.kmodels_jitter_w
            
    elif mode == "kruns": # This is for KRunsDataEnsembleStrategy
        # These args are SHARED (KRunsDataEnsembleStrategy accepts them)
        if args.kmodels_std_floor is not None:
            strategy_kwargs["std_floor"] = args.kmodels_std_floor
        if args.kmodels_std_rel_floor is not None:
            strategy_kwargs["std_rel_floor"] = args.kmodels_std_rel_floor
        # These args are UNIQUE to kruns
        if args.clone_alpha_mult_min is not None:
            strategy_kwargs["clone_alpha_mult_min"] = args.clone_alpha_mult_min
        if args.clone_alpha_mult_max is not None:
            strategy_kwargs["clone_alpha_mult_max"] = args.clone_alpha_mult_max
        # Note: kruns does NOT use jitter_u/v/w
        if args.wild_scale is not None:
            strategy_kwargs["wild_scale"] = args.wild_scale

    model.unc = make_strategy(mode if mode != "none" else None, **strategy_kwargs) if mode != "none" else None

    # Conformal controls on the model
    model.conformal_alpha = args.conf_alpha
    model.conformal_buffer_maxlen = args.conf_buf

    # Optional hetero multipliers (set as attributes if present)
    if mode == "hetero" and model.unc is not None:
        # only used if HeteroStrategy reads these attributes
        setattr(model.unc, "width_scale_nz", args.hetero_width_scale_nz)
        setattr(model.unc, "width_scale_z",  args.hetero_width_scale_z)

    t0 = time.time()
    model.fit_stream(T, dshape)
    t1 = time.time()

    # Summarize RMSE
    rmse_total = np.loadtxt(out_dir / "rmse_total_ssmf.txt")
    rmse_nz    = np.loadtxt(out_dir / "rmse_nonzeros_ssmf.txt")
    rmse_z     = np.loadtxt(out_dir / "rmse_zeros_ssmf.txt")
    

    def _maybe(path):
        p = out_dir / path
        return np.loadtxt(p) if p.exists() else None

    w_nz = _maybe("conformal_widths_nz.txt")
    c_nz = _maybe("conformal_coverages_nz.txt")
    w_z  = _maybe("conformal_widths_zero.txt")
    c_z  = _maybe("conformal_coverages_zero.txt")

    print(f"\n[{mode}] time={t1-t0:.2f}s  RMSE(total/nz/z)={rmse_total.mean():.4f}/{rmse_nz.mean():.4f}/{rmse_z.mean():.4f}")
    if c_nz is not None:
        warm = int(len(c_nz) * args.warm_frac)
        print(f"      COV_nz={np.nanmean(c_nz[warm:]):.5f}  W_nz(mean/med)={np.nanmean(w_nz[warm:]):.5f}/{np.nanmedian(w_nz[warm:]):.5f}")
    if c_z is not None:
        warm = int(len(c_z) * args.warm_frac)
        print(f"      COV_z ={np.nanmean(c_z[warm:]):.5f}  W_z (mean/med)={np.nanmean(w_z[warm:]):.6f}/{np.nanmedian(w_z[warm:]):.6f}")
    print(f"      -> outputs in {out_dir}")

def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("parquet", help="Parquet file with PU_idx/DO_idx/t_idx/trip_count")
    pa.add_argument("--out_root", default="out_real")
    pa.add_argument("--modes", default="homo,binwise,hetero", help="comma list among: none,homo,binwise,hetero")
    # model
    pa.add_argument("--periodicity", type=int, default=24)
    pa.add_argument("--n_components", type=int, default=10)
    pa.add_argument("--max_regimes", type=int, default=50)
    pa.add_argument("--max_iter", type=int, default=1)
    pa.add_argument("--learning_rate", type=float, default=0.2)
    pa.add_argument("--penalty", type=float, default=0.05)
    pa.add_argument("--float_cost", type=int, default=32)
    pa.add_argument("--update_freq", type=int, default=1)
    pa.add_argument("--init_cycles", type=int, default=3)
    # conformal controls
    pa.add_argument("--conf_alpha", type=float, default=0.10)
    pa.add_argument("--conf_buf", type=int, default=100_000)
    pa.add_argument("--warm_frac", type=float, default=0.30)
    # zero sampling
    pa.add_argument("--zero_samples", type=int, default=3000)
    pa.add_argument("--zero_top_frac", type=float, default=0.6)
    pa.add_argument("--zero_pool_mult", type=int, default=10)
    # homo
    pa.add_argument("--nz_samples", type=int, default=3000, help="How many nonzero residuals to add per slice for Homo (None disables)")
    pa.add_argument("--exante_tau", type=float, default=0.0, help="Forecast-time threshold: use zero interval if yhat <= tau, else nz interval")
    # binwise
    pa.add_argument("--bin_edges", type=str, default="auto", help="comma list or 'auto'")
    pa.add_argument("--bin_Nmin_zero", type=int, default=100)
    # hetero optional multipliers (used only if strategy reads them)
    pa.add_argument("--hetero_width_scale_nz", type=float, default=1.0)
    pa.add_argument("--hetero_width_scale_z",  type=float, default=1.0)
    # For 'bestn'
    pa.add_argument("--n_best", type=int, default=3,
                    help="Best-N regimes for ensemble uncertainty (bestn mode)")
    pa.add_argument("--ens_scale_nz", type=float, default=1.0,
                    help="Ensemble spread scale for nonzeros (bestn)")
    pa.add_argument("--ens_scale_z", type=float, default=1.0,
                    help="Ensemble spread scale for zeros (bestn)")
    pa.add_argument("--diversity_weight", type=float, default=None)     
    pa.add_argument("--norm_clip", type=float, default=None)
    pa.add_argument("--q_cap", type=float, default=None)                 
    # For 'kmodels'
    pa.add_argument("--kmodels_ens_scale_nz", type=float, default=None)
    pa.add_argument("--kmodels_ens_scale_z",  type=float, default=None)
    pa.add_argument("--kmodels_std_floor",    type=float, default=None)
    pa.add_argument("--kmodels_std_rel_floor",type=float, default=None)
    pa.add_argument("--kmodels_jitter_u",     type=float, default=None)
    pa.add_argument("--kmodels_jitter_v",     type=float, default=None)
    pa.add_argument("--kmodels_jitter_w",     type=float, default=None)
    # For 'kruns' 
    pa.add_argument("--clone_alpha_mult_min", type=float, default=None)
    pa.add_argument("--clone_alpha_mult_max", type=float, default=None)
    pa.add_argument("--wild_scale", type=float, default=None)
    # ------------------


    args = pa.parse_args()

    # Load parquet once
    df = (
        pd.read_parquet(args.parquet, columns=["PU_idx","DO_idx","t_idx","trip_count"])
          .astype({"PU_idx":"int32","DO_idx":"int32","t_idx":"int32","trip_count":"float32"})
          .sort_values("t_idx")
    )

    # Prepare bin edges
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    edges = None
    if "binwise" in modes:
        if args.bin_edges == "auto":
            edges = _auto_bin_edges_from_counts(df["trip_count"].values, num_bins=10)
        else:
            vals = [float(x) for x in args.bin_edges.split(",")]
            edges = np.array(vals, float)

    for mode in modes:
        run_one_mode(args, mode, df, edges)

if __name__ == "__main__":
    main()
