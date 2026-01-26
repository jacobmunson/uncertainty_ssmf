# uncertainty/conformal/hetero.py
from __future__ import annotations
import numpy as np
from collections import deque
from pathlib import Path
from typing import Optional, Iterable, Tuple
from .sampling import stratified_zero_indices
from .quantile import conformal_q


Triple = Tuple[int, int, float]
Triples = Iterable[Triple]

class HeteroStrategy:
    """
    Simple heteroskedastic conformal (baseline):
      - Fit |resid| ~ a + b * sqrt(yhat) on NONZERO residuals of current slice.
      - Conformal widths use normalized residuals & rolling quantile.
      - Zeros are sampled for evaluation only (don't affect scale fit).
    """

    def __init__(
        self,
        *,
        # zero sampling
        zero_samples_per_slice: int = 3000,
        zero_top_frac: float = 0.6,
        zero_pool_mult: int = 10,
        # scale-fitting knobs
        ridge: float = 1e-6,
        min_sigma: float = 1e-6,
        max_sigma: float | None = None,  
        # optional cosmetics / calibration
        width_scale_nz: float = 1.0,
        width_scale_z:  float = 1.0,
        # RNG
        rng_seed: int | None = None,
        **kwargs,
    ):
        import numpy as _np
        # zero-sampling
        self.m = int(zero_samples_per_slice)
        self.top_frac = float(zero_top_frac)
        self.pool_mult = int(zero_pool_mult)

        # scale fit
        self.ridge = float(ridge)
        self.min_sigma = float(min_sigma)
        self.max_sigma = None if max_sigma is None else float(max_sigma)

        # width scalers
        self.width_scale_nz = float(width_scale_nz)
        self.width_scale_z  = float(width_scale_z)

        # conformal meta
        self.alpha = 0.10
        self.model = None
        self._t = None

        # RNG
        self._rng_seed = rng_seed
        self.rng = _np.random.default_rng(rng_seed) if rng_seed is not None else _np.random.default_rng()

        # rolling buffers & logs (created in start)
        self._buf_nz = None
        self._buf_z  = None
        self.w_nz = None; self.c_nz = None
        self.w_z  = None; self.c_z  = None

        # non-oracle logs (created in start)
        self.w_no_nz = None; self.c_no_nz = None
        self.w_no_z  = None; self.c_no_z  = None

        # fixed forecast-only gate (no τ sweep): treat "very small" forecasts as zero-like
        # default matches Binwise smallest-bin edge on sqrt-scale
        self._small_edge = float(kwargs.get("small_edge", 1e-6))

        # scale parameters a,b (set/reset in start)
        self._a = 1e-3
        self._b = 0.0

    def start(self, *, alpha: float, buffer_maxlen: int, model=None):
        from collections import deque
        import numpy as _np
        self.alpha = float(alpha)
        self._target = 1.0 - self.alpha
        self._model = model

        self._buf_nz = deque(maxlen=int(buffer_maxlen))
        self._buf_z  = deque(maxlen=int(buffer_maxlen))

        self.w_nz, self.c_nz = [], []
        self.w_z,  self.c_z  = [], []

        # non-oracle logs
        self.w_no_nz, self.c_no_nz = [], []
        self.w_no_z,  self.c_no_z  = [], []

        # reset scale
        self._a = 1e-3
        self._b = 0.0

        # (re-)seed RNG if requested
        self.rng = _np.random.default_rng(self._rng_seed) if self._rng_seed is not None else _np.random.default_rng()


    def save(self, out: Path):
        np.savetxt(out / "conformal_widths_nz.txt",    np.nan_to_num(self.w_nz))
        np.savetxt(out / "conformal_coverages_nz.txt", np.nan_to_num(self.c_nz))
        np.savetxt(out / "conformal_widths_zero.txt",  np.nan_to_num(self.w_z))
        np.savetxt(out / "conformal_coverages_zero.txt", np.nan_to_num(self.c_z))
        np.savetxt(out / "hetero_params.txt", np.array([self._a, self._b]))

        # non-oracle (forecast-only routing) evaluation
        np.savetxt(out / "conformal_widths_nonoracle_nz.txt",      np.nan_to_num(self.w_no_nz))
        np.savetxt(out / "conformal_coverages_nonoracle_nz.txt",   np.nan_to_num(self.c_no_nz))
        np.savetxt(out / "conformal_widths_nonoracle_zero.txt",    np.nan_to_num(self.w_no_z))
        np.savetxt(out / "conformal_coverages_nonoracle_zero.txt", np.nan_to_num(self.c_no_z))

    # ---------- helpers ----------
    def _q(self, buf):
        return conformal_q(buf, self.alpha)

    def _is_small(self, yhat: np.ndarray) -> np.ndarray:
        # forecast-only gate on sqrt-scale (matches binwise convention)
        x = np.sqrt(np.clip(yhat, 0.0, None))
        return x <= self._small_edge

    def _sigma_from_yhat(self, yhat: np.ndarray) -> np.ndarray:
        sig = self._a + self._b * np.sqrt(np.clip(yhat, 0.0, None))
        sig = np.maximum(sig, self.min_sigma)
        if self.max_sigma is not None:
            sig = np.minimum(sig, self.max_sigma)
        return sig

    def _fit_scale_from_batch(self, yhat: np.ndarray, abs_resid: np.ndarray):
        """
        Fit (a,b) via 2x2 normal equations on current slice NZ residuals:
            y = a + b x, where x = sqrt(yhat)
        """
        if abs_resid.size == 0:
            return  # keep previous (a,b)

        x = np.sqrt(np.clip(yhat, 0.0, None)).astype(np.float64)
        y = abs_resid.astype(np.float64)

        n = float(x.size)
        sx = float(x.sum());  sy = float(y.sum())
        sxx = float((x*x).sum()); sxy = float((x*y).sum())

        S = np.array([[sxx + self.ridge, sx],
                      [sx,  n  + self.ridge]], dtype=np.float64)
        t = np.array([sxy, sy], dtype=np.float64)
        try:
            b_hat, a_hat = np.linalg.solve(S, t)
        except np.linalg.LinAlgError:
            return  # keep previous

        self._a = max(0.0, float(a_hat))
        self._b = max(0.0, float(b_hat))

    # ---------- main step ----------
    def step(self, F: np.ndarray, triples: Triples, d_shape: Tuple[int,int]):
        d0, d1 = d_shape

        # 1) read quantiles from PREVIOUS buffers (do not include this slice yet)
        q_nz = self._q(self._buf_nz)
        q_z  = self._q(self._buf_z)

        nz_cov = np.nan; nz_width_mean = np.nan
        z_cov  = np.nan; z_width_mean  = np.nan

        # non-oracle metrics
        nz_cov_no = np.nan; nz_width_mean_no = np.nan
        z_cov_no  = np.nan; z_width_mean_no  = np.nan        

        if triples:
            r, c, v = zip(*triples)
            r = np.asarray(r, np.int32); c = np.asarray(c, np.int32); v = np.asarray(v, float)
            yhat = F[r, c]

            # 2) size intervals using PREVIOUS scale (stored self._a, self._b)
            sigma_prev = self._sigma_from_yhat(yhat)  # uses stored (a,b)
            w = self.width_scale_nz * q_nz * sigma_prev
            lo = yhat - w; up = yhat + w
            nz_cov = float(((v >= lo) & (v <= up)).mean())
            nz_width_mean = float(np.mean(w))

            # non-oracle: choose q using forecast only (no label knowledge) ---
            use_qz = self._is_small(yhat)
            q_used = np.where(use_qz, q_z, q_nz).astype(float)
            w_no = self.width_scale_nz * q_used * sigma_prev
            lo_no = yhat - w_no; up_no = yhat + w_no
            nz_cov_no = float(((v >= lo_no) & (v <= up_no)).mean())
            nz_width_mean_no = float(np.mean(w_no))

        # zeros (sampled)
        zr, zc = stratified_zero_indices(F, triples, d_shape, self.m, self.top_frac, self.pool_mult, rng=self.rng)
        if zr is not None:
            yhat0 = F[zr, zc]
            sigma0_prev = self._sigma_from_yhat(yhat0)  # PREVIOUS scale
            w0 = self.width_scale_z * q_z * sigma0_prev
            lo0 = yhat0 - w0; up0 = yhat0 + w0
            z_cov = float(((0.0 >= lo0) & (0.0 <= up0)).mean())
            z_width_mean = float(np.mean(w0))

            # --- non-oracle: same forecast-only rule applied to zeros ---
            use_qz0 = self._is_small(yhat0)
            q_used0 = np.where(use_qz0, q_z, q_nz).astype(float)
            w0_no = self.width_scale_z * q_used0 * sigma0_prev
            lo0_no = yhat0 - w0_no; up0_no = yhat0 + w0_no
            z_cov_no = float(((0.0 >= lo0_no) & (0.0 <= up0_no)).mean())
            z_width_mean_no = float(np.mean(w0_no))

        # 3) AFTER evaluation: update scale and append normalized residuals for NEXT slice

        if triples:
            # fit NEW scale on this slice
            abs_res = np.abs(v - yhat)
            self._fit_scale_from_batch(yhat, abs_res)   # updates self._a, self._b

            # compute normalized residuals with the NEW scale
            sigma_new = self._sigma_from_yhat(yhat)
            norm_res = abs_res / np.maximum(sigma_new, 1e-12)
            for val in norm_res:
                self._buf_nz.append(float(val))

        if zr is not None:
            # normalized zeros: |yhat| / sigma(yhat) with the NEW scale
            sigma0_new = self._sigma_from_yhat(yhat0)
            z_norm = np.abs(yhat0) / np.maximum(sigma0_new, 1e-12)
            for val in z_norm:
                self._buf_z.append(float(val))

        # log & return
        self.w_nz.append(nz_width_mean); self.c_nz.append(nz_cov)
        self.w_z.append(z_width_mean);   self.c_z.append(z_cov)

        # non-oracle logs
        self.w_no_nz.append(nz_width_mean_no); self.c_no_nz.append(nz_cov_no)
        self.w_no_z.append(z_width_mean_no);   self.c_no_z.append(z_cov_no)

        return nz_width_mean, nz_cov, z_width_mean, z_cov