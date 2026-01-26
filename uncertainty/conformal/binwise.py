# uncertainty/conformal/binwise.py
from __future__ import annotations
import numpy as np
from collections import deque
from pathlib import Path
from typing import Optional, Iterable, Tuple, Sequence
from .sampling import stratified_zero_indices
from .quantile import conformal_q


Triple = Tuple[int, int, float]
Triples = Iterable[Triple]

class BinwiseStrategy:
    """
    Bin-wise conformal:
      - Partition by forecast magnitude using bin_edges (len B+1).
      - Separate buffers per bin for NZ and Z.
      - Width for a point = q_{1-alpha}(bin buffer) (scalar per bin).
    """

    def __init__(self, bin_edges=None, bin_Nmin_zero: int = 2000,
                 zero_samples_per_slice: int = 3000,
                 zero_top_frac: float = 0.6,
                 zero_pool_mult: int = 10,
                 rng_seed: int | None = None,
                 **kwargs):

        # allow late-binding from harness; default tiny->moderate; last is +inf on sqrt-scale
        if bin_edges is None:
            self.bin_edges = np.array([0.0, 1e-6, 5e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, np.inf], float)
        else:
            self.bin_edges = np.asarray(bin_edges, float)

        if not np.all(self.bin_edges[1:] >= self.bin_edges[:-1]):
            raise ValueError("bin_edges must be ascending.")
        self.bin_Nmin_zero = int(bin_Nmin_zero)

        self.m = int(zero_samples_per_slice)
        self.top_frac = float(zero_top_frac)
        self.pool_mult = int(zero_pool_mult)
        self.alpha = 0.10
        self.model = None
        self._t = None
        self._rng_seed = kwargs.get("rng_seed", None)   

        # buffers & logs created in start()
        self._bins_nz   = None
        self._bins_zero = None

        self.w_nz = None
        self.c_nz = None
        self.w_z = None
        self.c_z = None

    # ---------- lifecycle ----------
    def start(self, *, alpha: float, buffer_maxlen: int, model=None):
        self.alpha = float(alpha)
        self._target = 1.0 - self.alpha
        self._model = model

        self.B = len(self.bin_edges) - 1                    # <-- define B here

        # per-bin rolling buffers
        self._bins_nz   = [deque(maxlen=int(buffer_maxlen)) for _ in range(self.B)]
        self._bins_zero = [deque(maxlen=int(buffer_maxlen)) for _ in range(self.B)]

        # rng is optional; allow None for unpredictable
        self.rng = np.random.default_rng(self._rng_seed) if self._rng_seed is not None else np.random.default_rng()

        # oracle width and coverage storage
        self.w_nz, self.c_nz = [], []
        self.w_z,  self.c_z  = [], []

        # non-oracle width and coverage storage
        self.w_no_nz, self.c_no_nz = [], []
        self.w_no_z,  self.c_no_z  = [], []


    def save(self, out: Path):
        # save oracle width and coverages
        np.savetxt(out / "conformal_widths_nz.txt",      np.nan_to_num(self.w_nz))
        np.savetxt(out / "conformal_coverages_nz.txt",   np.nan_to_num(self.c_nz))
        np.savetxt(out / "conformal_widths_zero.txt",    np.nan_to_num(self.w_z))
        np.savetxt(out / "conformal_coverages_zero.txt", np.nan_to_num(self.c_z))

        # save non-oracle width and coverages
        np.savetxt(out / "conformal_widths_nonoracle_nz.txt",      np.nan_to_num(self.w_no_nz))
        np.savetxt(out / "conformal_coverages_nonoracle_nz.txt",   np.nan_to_num(self.c_no_nz))
        np.savetxt(out / "conformal_widths_nonoracle_zero.txt",    np.nan_to_num(self.w_no_z))
        np.savetxt(out / "conformal_coverages_nonoracle_zero.txt", np.nan_to_num(self.c_no_z))

        # save bin edges for records
        np.save(out / "bin_edges.npy", self.bin_edges)

    def _q(self, buf):
        return conformal_q(buf, self.alpha)

    def _bin_index(self, yhat: np.ndarray) -> np.ndarray:
        x = np.sqrt(np.clip(yhat, 0.0, None))          # bin on sqrt scale
        b = np.digitize(x, self.bin_edges[1:-1], right=True)
        return np.clip(b, 0, self.B - 1).astype(int)

    # ---------- main step ----------
    def step(self, F: np.ndarray, triples: Triples, d_shape: Tuple[int,int]):
        b_star = 0  # fixed gate: only smallest bin uses zero-calibrator

        # 1) read per-bin quantiles from PREVIOUS buffers
        q_nz_bins   = np.array([self._q(self._bins_nz[i])   for i in range(self.B)], dtype=float)
        q_zero_bins = np.array([self._q(self._bins_zero[i]) for i in range(self.B)], dtype=float)


        nz_cov_no = np.nan; nz_w_mean_no = np.nan
        z_cov_no  = np.nan; z_w_mean_no  = np.nan
        nz_cov = np.nan; nz_w_mean = np.nan


        # ----- Nonzeros: evaluate using OLD q's -----
        if triples:
            r, c, v = zip(*triples)
            r = np.asarray(r, np.int32); c = np.asarray(c, np.int32); v = np.asarray(v, float)
            yhat = F[r, c]
            b = self._bin_index(yhat)
            w = q_nz_bins[b].astype(float)
            bad = (~np.isfinite(w)) | (w <= 0.0)
            if bad.any():
                qnz_global = float(np.median(q_nz_bins[q_nz_bins > 0])) if np.any(q_nz_bins > 0) else 1e-6
                w[bad] = qnz_global
            lo = yhat - w; up = yhat + w
            nz_cov = float(((v >= lo) & (v <= up)).mean()) if v.size else np.nan
            nz_w_mean = float(w.mean()) if w.size else np.nan

            # route by bin only (no label): small bins use q_zero, others q_nz
            # non-oracle width
            w_no = np.where(b <= b_star, q_zero_bins[b], q_nz_bins[b]).astype(float)


            bad = (~np.isfinite(w_no)) | (w_no <= 0.0)
            if bad.any():
                # conservative fallback: median of available positive widths from both arrays
                pool = np.concatenate([q_nz_bins[q_nz_bins > 0], q_zero_bins[q_zero_bins > 0]])
                w_global = float(np.median(pool)) if pool.size else 1e-6
                w_no[bad] = w_global

            lo_no = yhat - w_no
            up_no = yhat + w_no
            nz_cov_no = float(((v >= lo_no) & (v <= up_no)).mean()) if v.size else np.nan
            nz_w_mean_no = float(w_no.mean()) if w_no.size else np.nan


        # ----- Zeros: evaluate using OLD q's -----
        z_cov = np.nan; z_w_mean = np.nan
        zr = zc = None
        if self.m > 0:
            zr, zc = stratified_zero_indices(F, triples, d_shape, self.m, self.top_frac, self.pool_mult, rng=self.rng)
        if zr is not None and zc is not None and zr.size:
            yhat0 = F[zr, zc]
            b0 = self._bin_index(yhat0)
            w0 = q_zero_bins[b0].astype(float)
            if (~np.isfinite(w0)).any() or (w0 <= 0.0).any():
                qz_global = float(np.median(q_zero_bins[q_zero_bins > 0])) if np.any(q_zero_bins > 0) else 1e-6
                w0[(~np.isfinite(w0)) | (w0 <= 0.0)] = qz_global
            # thin-bin guard
            Nmin = self.bin_Nmin_zero
            cnt_zero = np.array([len(self._bins_zero[i]) for i in range(self.B)], dtype=float)
            thin = cnt_zero[b0] < Nmin
            if np.any(thin):
                qz_global = float(np.median(q_zero_bins[q_zero_bins > 0])) if np.any(q_zero_bins > 0) else 1e-6
                w0[thin] = qz_global

            lo0 = yhat0 - w0; up0 = yhat0 + w0
            z_cov = float(((0.0 >= lo0) & (0.0 <= up0)).mean()) if w0.size else np.nan
            z_w_mean = float(w0.mean()) if w0.size else np.nan

            # non-oracle width
            w0_no = np.where(b0 <= b_star, q_zero_bins[b0], q_nz_bins[b0]).astype(float)

            bad = (~np.isfinite(w0_no)) | (w0_no <= 0.0)
            if bad.any():
                pool = np.concatenate([q_nz_bins[q_nz_bins > 0], q_zero_bins[q_zero_bins > 0]])
                w_global = float(np.median(pool)) if pool.size else 1e-6
                w0_no[bad] = w_global


            # keep the same thin-bin guard idea if you want (optional but good):
            #Nmin = self.bin_Nmin_zero
            #cnt_zero = np.array([len(self._bins_zero[i]) for i in range(self.B)], dtype=float)
            #thin = cnt_zero[b0] < Nmin
            #if np.any(thin):
                # if zero bins are thin, fall back to nz widths for those bins (conservative)
            #    w0_no[thin] = q_nz_bins[b0][thin]

            lo0_no = yhat0 - w0_no
            up0_no = yhat0 + w0_no
            z_cov_no = float(((0.0 >= lo0_no) & (0.0 <= up0_no)).mean()) if w0_no.size else np.nan
            z_w_mean_no = float(w0_no.mean()) if w0_no.size else np.nan


        # 3) AFTER evaluation: append current slice scores to bins
        if triples:
            abs_res = np.abs(v - yhat)
            for val, bi in zip(abs_res, b):
                self._bins_nz[int(bi)].append(float(val))
        if zr is not None and zc is not None and zr.size:
            abs_res0 = np.abs(yhat0)  # truth=0
            b0_ = self._bin_index(yhat0)
            for val, bi in zip(abs_res0, b0_):
                self._bins_zero[int(bi)].append(float(val))

        # log & return
        self.w_nz.append(nz_w_mean); self.c_nz.append(nz_cov)
        self.w_z.append(z_w_mean);   self.c_z.append(z_cov)

        # non-oracle results
        self.w_no_nz.append(nz_w_mean_no); self.c_no_nz.append(nz_cov_no)
        self.w_no_z.append(z_w_mean_no);   self.c_no_z.append(z_cov_no)

        return nz_w_mean, nz_cov, z_w_mean, z_cov