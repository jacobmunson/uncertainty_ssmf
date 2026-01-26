# uncertainty/conformal/homo.py
from __future__ import annotations
import numpy as np
from collections import deque
from pathlib import Path
from typing import Optional
from .base import Triples
from .sampling import stratified_zero_indices
from .quantile import conformal_q


class HomoStrategy:
    def __init__(
        self,
        zero_samples_per_slice: int = 3000,
        zero_top_frac: float = 0.6,
        zero_pool_mult: int = 10,
        nz_samples_per_slice: Optional[int] = 3000, 
        **kwargs,  
    ):
        self.m = int(zero_samples_per_slice)
        self.top_frac = float(zero_top_frac)
        self.pool_mult = int(zero_pool_mult)
        self.alpha = 0.10
        self.model = None
        self._t = None
        self.nz_m = None if nz_samples_per_slice is None else int(nz_samples_per_slice)

        # ex-ante selection rule: use q_z when yhat <= tau, else q_nz
        self.exante_tau = float(kwargs.get("exante_tau", 0.0))

        self._rng_seed = kwargs.get("rng_seed", None)   # <-- add this

    def start(self, *, alpha: float, buffer_maxlen: int, model=None):
        self.rng = np.random.default_rng(self._rng_seed)

        self.alpha = alpha
        self.nz = deque(maxlen=buffer_maxlen)
        self.ze = deque(maxlen=buffer_maxlen)
        self.w_nz, self.c_nz, self.w_z, self.c_z = [], [], [], []

        self.w_ex_nz, self.c_ex_nz = [], []
        self.w_ex_z, self.c_ex_z = [], []


        self._model = model


    def _q(self, buf):
        return conformal_q(buf, self.alpha)
       
    def step(self, F: np.ndarray, triples: Triples, d_shape):
        triples = list(triples) if triples is not None else []

        # 1) read quantiles from PREVIOUS buffers
        q_nz = self._q(self.nz)
        q_z  = self._q(self.ze)
        tau = self.exante_tau

        # 2) form intervals & evaluate coverage for THIS slice
        if triples:
            r, c, v = zip(*triples)
            r = np.asarray(r, np.int32); c = np.asarray(c, np.int32); v = np.asarray(v, float)
            yhat = F[r, c]
            lo = np.maximum(yhat - q_nz, 0.0); up = yhat + q_nz
            cov_nz = ((v >= lo) & (v <= up)).mean(); avg_w_nz = float(q_nz)
        else:
            cov_nz = np.nan; avg_w_nz = np.nan

        zr, zc = stratified_zero_indices(F, triples, d_shape, self.m, self.top_frac, self.pool_mult, rng=self.rng)
        if zr is not None and zc is not None and np.size(zr) > 0:
            y0 = F[zr, zc]

            # ---- Oracle zeros (uses q_z because truth is zero) ----
            lo0 = np.maximum(y0 - q_z, 0.0); up0 = y0 + q_z
            cov_z = ((0.0 >= lo0) & (0.0 <= up0)).mean()
            avg_w_z = float(q_z)

            # ---- Ex-ante zeros (forecast-time rule uses only y0) ----
            w_ex0 = np.where(y0 <= tau, q_z, q_nz).astype(float)
            lo_ex0 = np.maximum(y0 - w_ex0, 0.0)
            up_ex0 = y0 + w_ex0
            cov_ex0 = float(((0.0 >= lo_ex0) & (0.0 <= up_ex0)).mean()) if y0.size else np.nan
            avg_w_ex0 = float(np.mean(w_ex0)) if w_ex0.size else np.nan

        else:
            cov_z = np.nan; avg_w_z = np.nan
            cov_ex0 = np.nan; avg_w_ex0 = np.nan

        # ---- Ex-ante (forecast-time) coverage on realized nonzero tuples ----
        # Use only yhat to decide which interval family to apply.
        if triples:
            w_ex = np.where(yhat <= tau, q_z, q_nz).astype(float)
            lo_ex = np.maximum(yhat - w_ex, 0.0)
            up_ex = yhat + w_ex
            cov_ex = float(((v >= lo_ex) & (v <= up_ex)).mean()) if v.size else np.nan
            avg_w_ex = float(np.mean(w_ex)) if w_ex.size else np.nan
        else:
            cov_ex = np.nan
            avg_w_ex = np.nan

        # 3) AFTER evaluation: append THIS slice's scores to buffers
        if triples:
            e = np.abs(v - yhat);           
            if self.nz_m is not None and e.size > self.nz_m:
                idx = self.rng.choice(e.size, size=self.nz_m, replace=False)
                e = e[idx]
            self.nz.extend(map(float, e))
        if zr is not None and zc is not None and np.size(zr) > 0:
            e0 = np.abs(y0); self.ze.extend(map(float, e0))

        self.w_nz.append(avg_w_nz); self.c_nz.append(cov_nz)
        self.w_z.append(avg_w_z);   self.c_z.append(cov_z)

        self.w_ex_nz.append(avg_w_ex);  self.c_ex_nz.append(cov_ex)
        self.w_ex_z.append(avg_w_ex0);  self.c_ex_z.append(cov_ex0)

        return avg_w_nz, cov_nz, avg_w_z, cov_z


    def save(self, out: Path):
        # save oracle width and coverages
        np.savetxt(out/"conformal_widths_nz.txt",   np.nan_to_num(self.w_nz))
        np.savetxt(out/"conformal_widths_zero.txt", np.nan_to_num(self.w_z))
        np.savetxt(out / "conformal_coverages_nz.txt",   np.asarray(self.c_nz, float))
        np.savetxt(out / "conformal_coverages_zero.txt", np.asarray(self.c_z,  float))

        # save non-oracle width and coverages
        np.savetxt(out/"conformal_widths_exante_nz.txt",     np.nan_to_num(self.w_ex_nz))
        np.savetxt(out/"conformal_coverages_exante_nz.txt",  np.asarray(self.c_ex_nz, float))
        np.savetxt(out/"conformal_widths_exante_z.txt",      np.nan_to_num(self.w_ex_z))
        np.savetxt(out/"conformal_coverages_exante_z.txt",   np.asarray(self.c_ex_z, float))

        # save tau value for records
        np.savetxt(out / "exante_tau.txt", np.array([self.exante_tau]))