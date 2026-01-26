"""
Shifting Seasonal Matrix Factorization on a parquet tuple stream.
Keeps features from the original dense version. 
Processes data as input tuples vs. dense (explicit 0 representations) tensor.

To run:
python ssmf_tuples.py taxi_yellow_green_rideshare_distinct_march_to_apr2020_triplets.parquet
"""
from __future__ import annotations
import argparse, time, warnings
from collections import deque, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Deque, Iterator, List, Sequence, Tuple
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import norm
from tqdm import tqdm
import numba
from joblib import Parallel, delayed
import ncp, utils   

from uncertainty.conformal import make_strategy


USE_NUMBA = True


def rmse_components_tuple(forecast: np.ndarray,
                          triples: List[Tuple[int, int, float]],
                          d_shape: Tuple[int, int]):
    """
    Compare dense forecast (d0×d1) with sparse ground-truth tuples.
    Returns (rmse_total, rmse_zeros, rmse_nonzeros).    <- more nuanced view of RMSE
    """
    d0, d1 = d_shape
    total_elements = d0 * d1
    
    # 1. Calculate error on non-zero elements
    sse_nonzero = 0.0
    num_nonzero = len(triples)
    if num_nonzero > 0:
        r, c, v = zip(*triples)
        r, c = np.array(r), np.array(c) # Convert to numpy arrays for indexing
        v = np.array(v)
        
        diff_nonzero = forecast[r, c] - v
        sse_nonzero = np.sum(diff_nonzero ** 2)
        rmse_nonzeros = np.sqrt(sse_nonzero / num_nonzero)
    else:
        rmse_nonzeros = 0.0
        
    # 2. Calculate error on zero elements
    num_zeros = total_elements - num_nonzero
    sse_zeros = 0.0
    if num_zeros > 0:
        # The error on a zero element is (0 - forecast_value)^2 = forecast_value^2
        # Sum of all squared forecast values
        sse_forecast_total = np.sum(forecast ** 2)
        
        # Sum of squared forecast values ONLY at the non-zero locations
        sse_forecast_at_nonzero = np.sum(forecast[r, c] ** 2) if num_nonzero > 0 else 0
        
        # By subtraction, we get the sum of squared forecast values at zero locations
        sse_zeros = sse_forecast_total - sse_forecast_at_nonzero
        rmse_zeros = np.sqrt(sse_zeros / num_zeros)
    else:
        rmse_zeros = 0.0
        
    # 3. Calculate total RMSE
    total_sse = sse_nonzero + sse_zeros
    rmse_total = np.sqrt(total_sse / total_elements)
    
    return rmse_total, rmse_zeros, rmse_nonzeros



class SSMF:
    """
    Shifting Seasonal Matrix Factorization for sparse tuple streams.
    Includes full regime creation and online update logic.
    """
    def __init__(self,
                 triplet_dict, periodicity, n_components,
                 max_regimes=100, epsilon=1e-12,
                 alpha=0.1, beta=0.05, max_iter=5, update_freq=1,
                 init_cycles=3, float_cost=32,
                 uncertainty_mode: str = "binwise",
                 **unc_kwargs): 

        assert periodicity  > 0 and n_components > 1
        assert max_regimes  > 0 and init_cycles  > 1

        self.triplet_dict = triplet_dict
        self.s = periodicity
        self.k = n_components
        self.r = max_regimes
        self.g = 1  # of current regimes

        self.eps = epsilon
        self.alpha = alpha
        self.beta = beta
        self.init_cycles = init_cycles
        self.max_iter = max_iter
        self.update_freq = update_freq
        self.float_cost = float_cost
        # These will be set by initialize()
        self.d = (0,0)
        self.n = 0
        self.U = []
        self.W = np.array([])
        self.R = np.array([])
        self.output_dir = "out"

        # --- conformal defaults (always present) ---
        self.conformal_alpha = 0.10
        self.conformal_buffer_maxlen = 100_000
        
        # per-slice logs (created once; cleared in fit_stream)
        self.pi_widths_nz = []
        self.pi_coverages_nz = []
        self.pi_widths_zero = []
        self.pi_coverages_zero = []


        # build uncertainty strategy
        self.unc = None
        if uncertainty_mode is not None:
            # one-line factory — forward everything
            self.unc = make_strategy(uncertainty_mode, **unc_kwargs)
    

    def initialize(self, X_window: np.ndarray, n_total: int):
        """
        Initializes factors from a dense tensor built from the first few seasons.
        """
        self.d = X_window.shape[:-1]
        self.n = n_total

        self.U = [np.zeros((i, self.k)) for i in self.d]
        self.W = np.zeros((self.r, self.s + self.n, self.k))
        self.R = np.zeros(self.n, dtype=int)

        factor = ncp.ncp(X_window, self.k, maxit=3, verbose=False)      # <- NCP call
        self.W[0, :self.s] = factor[-1] # Only initialize for regime 0

        # Normalization
        for i in range(len(self.d)):
            weights = np.sqrt(np.sum(factor[i] ** 2, axis=0))
            weights[weights == 0] = self.eps # Avoid division by zero
            self.U[i] = factor[i] @ np.diag(1 / weights)
            self.W[0, :self.s] = self.W[0, :self.s] @ np.diag(weights)

    def regime_generation_tuples(self, window_triples: Sequence[List[Tuple]], t: int, ridx: int, max_iter: int):
        """
        Creates a candidate new regime by fitting new factors to the current data window.
        This is the sparse-data equivalent of the original 'regime_generation'.
        """
        # 1. Initialize new factors by copying current
        #    (U_new is a list [U0, U1], W_new is a (s, k) numpy array)
        U_new = deepcopy(self.U)
        W_new = self.W[ridx, t - self.s + 1 : t + 1].copy()

        # 2. Refine candidate factors over entire window
        # Pre-compute Gram matrices once for this pass       
        if USE_NUMBA:
            for _ in range(max_iter):
                for i, triples_in_slice in enumerate(window_triples):
                    if triples_in_slice:
                        r, c, v = zip(*triples_in_slice)
                        U_new[0], U_new[1], W_new[i] = self.apply_grad_numba_parallel(
                            U_new[0], U_new[1], W_new[i],
                            np.array(r, dtype=np.int32), np.array(c, dtype=np.int32), np.array(v, float),
                            alpha=0.5, eps=self.eps
                        )
        else: 
            gram1 = U_new[1].T @ U_new[1]
            gram0 = U_new[0].T @ U_new[0]
            for i, triples_in_slice in enumerate(window_triples):
                U_new[0], U_new[1], W_new[i] = self.apply_grad_sparse(
                    U_new, W_new[i], triples_in_slice, self.d, alpha=0.5, eps=self.eps, gram_matrices=(gram1, gram0) 
                )

        # 3. Calculate the total cost of this newly refined regime
        # a) Coding cost (using the probabilistic method)
        coding_cost = 0.0
        for i, triples_in_slice in enumerate(window_triples):
            u, v, w = U_new[0], U_new[1], W_new[i]
            coding_cost += utils.coding_cost_tuples_probabilistic(triples_in_slice, u, v, w, self.d)

        # b) Model cost (the cost of storing the new W factor)
        model_cost = utils.compute_model_cost(W_new, self.float_cost, self.eps)
        total_cost = coding_cost + model_cost

        return total_cost, U_new, W_new

    def fit_stream(self, n_total: int, d_shape: Tuple[int, int]):
        """
        Main online loop, including regime selection/generation logic.
        """
        s = self.s
        forecasts, rmse_total_list, rmse_zeros_list, rmse_nonzeros_list = [], [], [], []

        # 1. Initialization 
        print("Initializing model from first cycles...")
        init_end_t = s * self.init_cycles
        init_window_data = self.make_window(init_end_t - 1, init_end_t) # A single large window
        
        init_tensor = np.zeros((*d_shape, init_end_t))                  # Mirror original SSMF initialization logic
        for i, (_, triples) in enumerate(init_window_data):
            if triples:
                r, c, v = zip(*triples)
                init_tensor[r, c, i] = v
        
        X_fold = [init_tensor[..., i*s:(i+1)*s] for i in range(self.init_cycles)]
        X_fold_avg = np.array(X_fold).sum(axis=0) / self.init_cycles
        self.initialize(X_fold_avg, n_total)

        # 1.5. Uncertainty
        # ---- Uncertainty setup ----
        # Ensure conformal defaults exist (they will, thanks to __init__)
        # Clear per-run logs
        self.pi_widths_nz.clear()
        self.pi_coverages_nz.clear()
        self.pi_widths_zero.clear()
        self.pi_coverages_zero.clear()
        
        # Start the strategy
        if self.unc is not None:
             self.unc.start(alpha=self.conformal_alpha,
                            buffer_maxlen=self.conformal_buffer_maxlen,
                            model=self)

        # --- OPTIONAL: enable multi-horizon homoskedastic CP ---
        # --- Multi-horizon (homoskedastic) setup ---
        # Number of horizons to issue per slice. Define self.mh_horizons upstream (CLI/ctor), or set it on the model.
        H = getattr(self, "mh_horizons", 0)

        if H > 0:
            from uncertainty.conformal.horizon_homo import HorizonHomo

            # Pull parameters from the already-constructed uncertainty strategy (same ones homo/binwise/hetero use)
            zero_m   = getattr(self.unc, "m", 2000)                 
            top_frac = getattr(self.unc, "top_frac", 0.02)
            pool_mult= getattr(self.unc, "pool_mult", 10)

            # Use existing conformal alpha (already passed self.unc.start(...))
            alpha = self.conformal_alpha

            # Optional RNG on the model if present; otherwise HorizonHomo will seed internally
            rng = getattr(self, "rng", None)

            self._mh = HorizonHomo(alpha=alpha,
                                zero_m=zero_m,
                                top_frac=top_frac,
                                pool_mult=pool_mult,
                                rng=rng)
        else:
            self._mh = None


        
        # 2. Online Loop
        # for t in tqdm(range(s, n_total - 1), unit="slice", desc="Streaming fit"):  # if you want a progress bar
        for t in range(s, n_total - 1): 
            self._last_t = t
            self.W[:, t] = self.W[:, t - s] # Seasonal weights

            window_data = self.make_window(t, s)
            window_triples = [triples for _, triples in window_data]
            
            # a) Cost of using an existing regime
            cost1, ridx1 = self.regime_selection_vectorized(window_triples, t)

            # b) Cost of generating a new one (optional)
            cost2 = np.inf
            if t % self.update_freq == 0:
                cost2, Unew, Wnew = self.regime_generation_tuples(window_triples, t, ridx1, self.max_iter)

            # c) Create new regime or use existing?
            if cost1 + self.beta * cost1 < cost2:           # Use existing regime
                self.R[t] = ridx1
            else:                                           # Create a new regime
                if self.g < self.r:
                    #print(f"\nTime {t}: New regime {self.g} created.")
                    self.R[t] = self.g
                    self.U = Unew
                    self.W[self.g, t-s+1 : t+1] = Wnew
                    self.g += 1
                else:
                    # Max regimes reached, fall back to best existing
                    self.R[t] = ridx1
                    #if not self.g == 1:
                        #warnings.warn(f"Time {t}: # of regimes exceeded the limit ({self.r})")
            
            # d) Final gradient update on the current time slice using the chosen regime
            if USE_NUMBA:
                final_ridx = self.R[t]
                current_wt = self.W[final_ridx, t]
                current_triples = window_triples[-1]

                if current_triples: # Only update if there's data (there should be)
                    r, c, v = zip(*current_triples)
                    self.U[0], self.U[1], self.W[final_ridx, t] = self.apply_grad_numba(
                        self.U[0], self.U[1], current_wt,
                        np.array(r, dtype=np.int32), np.array(c, dtype=np.int32), np.array(v),
                        alpha=self.alpha, eps=self.eps
                    )
            else:
                final_ridx = self.R[t]
                current_wt = self.W[final_ridx, t]
                current_triples = window_triples[-1]
                self.U[0], self.U[1], self.W[final_ridx, t] = self.apply_grad_sparse(       # <- do the gradient update
                    self.U, current_wt, current_triples, d_shape, self.alpha, self.eps
                )

            # --- 1.5: Multi-horizon snapshot (SSMF state at time t, AFTER updating slice t) ---
            if self._mh is not None:
                # A light snapshot using existing arrays
                snap = SSMFStateSnapshot(
                    U=self.U[0].copy(),
                    V=self.U[1].copy(),
                    regime_id=int(self.R[t]),
                    W_view=self.W,                 
                    period=self.s                   
                )

                # --- 1.6: Issue H-step intervals at time t (pin decisions) ---
                # This reads past-only q_h and stores (q_h, zero idx, snapshot) in a pending registry keyed by target time.
                self._mh.issue(t=t, H=H, d_shape=d_shape, ssmf_state=snap)


            # 3. Forecast and Evaluate RMSE 
            forecast_t = self.forecast(self.R[t], t, steps_ahead=1)
            forecasts.append(forecast_t)

            next_triples = self.triplet_dict.get(t + 1, [])
            total, zeros, nonzeros = rmse_components_tuple(forecast_t, next_triples, d_shape)
            rmse_total_list.append(total)
            rmse_zeros_list.append(zeros)
            rmse_nonzeros_list.append(nonzeros)

            # --- 3.5: Multi-horizon observe (evaluate pinned intervals whose target_t == t+1) ---
            if self._mh is not None:
                self._mh.observe(target_t=t+1, triples=next_triples, d_shape=d_shape)


            if self.unc is not None:
                w_nz, c_nz, w_z, c_z = self.unc.step(forecast_t, next_triples, d_shape)
                self.pi_widths_nz.append(w_nz);     self.pi_coverages_nz.append(c_nz)
                self.pi_widths_zero.append(w_z);    self.pi_coverages_zero.append(c_z)
                
        #------- 4. Save results------------------------------------
        if self.unc is not None:
            self.unc.save(Path(self.output_dir))

        self.save_results(forecasts, rmse_total_list, rmse_zeros_list, rmse_nonzeros_list)

    def regime_selection_vectorized(self, window_triples: Sequence[List[Tuple]], t: int):
        """
        Vectorization to make per regime calculation fast.
        joblib to run the work for multiple regimes in parallel.
        """
        # 1. Aggregate all non-zero data from the entire window.
        # This setup is done once before we evaluate any regimes.
        all_r, all_c, all_v, all_t_local = [], [], [], []
        for i, triples in enumerate(window_triples):
            if not triples: continue
            r, c, v = zip(*triples)
            all_r.extend(r); all_c.extend(c); all_v.extend(v)
            all_t_local.extend([i] * len(r))

        if not all_r: return 0.0, 0
        all_r, all_c, all_v, all_t_local = map(np.array, [all_r, all_c, all_v, all_t_local])
        U0, U1 = self.U

        # 2. Work for a single regime.
        def _cost_for_one_regime(ridx):
            W_window = self.W[ridx, t - self.s + 1 : t + 1]
            
            # Vectorized prediction and cost calculation
            U0_rows = U0[all_r]
            U1_rows = U1[all_c]
            W_rows = W_window[all_t_local]
            v_preds = np.einsum('ij,ij->i', U0_rows, W_rows * U1_rows)
            errors = all_v - v_preds
            
            if errors.size < 2: return 0.0
            
            error_mean = errors.mean()
            error_std = errors.std()
            if error_std < 1e-9: error_std = 1e-9
                
            logprob = norm.logpdf(errors, loc=error_mean, scale=error_std)
            return -1 * logprob.sum() / np.log(2.0)

        # 3. Use parallel resources?
        if self.g <= 2: 
            all_costs = [_cost_for_one_regime(r) for r in range(self.g)]
        else:                                           
            # When g > 2, use joblib to run in parallel.
            all_costs = Parallel(n_jobs=-1, backend="threading")(
                delayed(_cost_for_one_regime)(r) for r in range(self.g)
            )

        E = np.array(all_costs)
        best_idx = np.argmin(E)
        return E[best_idx], best_idx

    def regime_selection_vectorized(self, window_triples, t):
        # pre-aggregate indices 
        U0, U1 = self.U

        def _cost_for_one_regime(ridx):
            W_window = self.W[ridx, t - self.s + 1 : t + 1]
            coding_cost = 0.0
            for i, triples in enumerate(window_triples):
                if not triples: 
                    continue
                coding_cost += utils.coding_cost_tuples_probabilistic(
                    triples, U0, U1, W_window[i], self.d
                )
            model_cost = utils.compute_model_cost(W_window, self.float_cost, self.eps)
            return coding_cost + model_cost

        if not any(len(tr) for tr in window_triples):
            return float("inf"), 0

        if self.g <= 2:
            all_costs = [_cost_for_one_regime(r) for r in range(self.g)]
        else:
            all_costs = Parallel(n_jobs=-1, backend="threading")(
                delayed(_cost_for_one_regime)(r) for r in range(self.g)
            )

        E = np.array(all_costs)
        best_idx = int(np.argmin(E))
        return float(E[best_idx]), best_idx
    
    def regime_selection_vectorized(self, window_triples, t: int):
        """
        Fast regime selection:
        - Vectorized residuals under a per-regime Normal(mean, std)
        - Adds model-cost on W window (MDL parity with generation)
        - No joblib (overhead > benefit when arrays are large)
        """

        # 1) Aggregate indices once for the whole window
        all_r, all_c, all_v, all_t_local = [], [], [], []
        for i, triples in enumerate(window_triples):
            if not triples:
                continue
            r, c, v = zip(*triples)
            all_r.extend(r); all_c.extend(c); all_v.extend(v)
            all_t_local.extend([i] * len(r))

        # Empty window: return neutral-but-large cost so caller "keeps" regime
        if not all_r:
            return float("inf"), 0

        # 2) Pack to arrays (prefer float32 to cut bandwidth)
        all_r = np.asarray(all_r, dtype=np.int32)
        all_c = np.asarray(all_c, dtype=np.int32)
        all_v = np.asarray(all_v, dtype=np.float32)
        all_t_local = np.asarray(all_t_local, dtype=np.int32)

        U0, U1 = self.U
        # Precompute Z = U0_rows * U1_rows (n_obs, k)
        U0_rows = U0[all_r]                   # (n_obs, k)
        U1_rows = U1[all_c]                   # (n_obs, k)
        Z = U0_rows * U1_rows                 # (n_obs, k)

        best_cost = float("inf")
        best_idx = 0

        # 3) Evaluate each regime
        for ridx in range(self.g):
            # W window for this regime: shape (s, k)
            W_window = self.W[ridx, t - self.s + 1 : t + 1]     # (s, k)

            # Gather W rows per observation’s local time index
            W_rows = W_window[all_t_local]                      # (n_obs, k)

            # Predictions: v_hat_i = sum_f Z[i,f] * W_rows[i,f]
            v_hat = np.einsum('ik,ik->i', Z, W_rows, optimize=True)

            # Residuals and Gaussian NLL in bits
            err = all_v - v_hat
            # guard tiny std to avoid -inf
            mu = err.mean(dtype=np.float64)
            sd = err.std(dtype=np.float64)
            if sd < 1e-9:
                sd = 1e-9
            # logpdf in nats: 0.5*log(2πσ^2) + (e-μ)^2/(2σ^2)
            n = err.size
            nll_nats = 0.5 * n * (np.log(2.0 * np.pi) + 2.0 * np.log(sd)) + np.sum((err - mu) ** 2) / (2.0 * sd * sd)
            nll_bits = nll_nats / np.log(2.0)

            # Add model cost for MDL parity (crucial!)
            mdl = nll_bits + utils.compute_model_cost(W_window, self.float_cost, self.eps)

            if mdl < best_cost:
                best_cost = mdl
                best_idx = ridx

        return float(best_cost), int(best_idx)

    @staticmethod
    def apply_grad_sparse(U_in, wt, triples, d_shape, alpha, eps, gram_matrices=None):
        U0, U1 = U_in[0].copy(), U_in[1].copy()
        k = U0.shape[1]

        # Data terms: Xt * (U * wt)  and Xt^T * (U * wt)
        if triples:
            r, c, v = zip(*triples)
            Xt_csr = sparse.coo_matrix((v, (r, c)), shape=d_shape).tocsr()
            grad0 = Xt_csr.dot(U1 * wt)
            grad1 = Xt_csr.T.dot(U0 * wt)
        else:
            grad0 = np.zeros_like(U0)
            grad1 = np.zeros_like(U1)

        # Weighted Gram terms: D (U^T U) D  == (U*wt)^T (U*wt)
        if gram_matrices is None:
            G1 = (U1 * wt).T @ (U1 * wt)
            G0 = (U0 * wt).T @ (U0 * wt)
        else:
            # If external grams are supplied, they should be the unweighted ones.
            gram1, gram0 = gram_matrices
            # Apply weighting on both sides: D gram D
            W = np.diag(wt)  # small (k×k); or rebuild as (wt[:,None]*gram*wt[None,:])
            G1 = W @ gram1 @ W
            G0 = W @ gram0 @ W

        grad0 -= U0 @ G1
        grad1 -= U1 @ G0

        # Step + column renorm (absorbing scales into wt as before)
        wt_new = wt.copy()
        for Umat, G in ((U0, grad0), (U1, grad1)):
            gnorm = np.linalg.norm(G)
            if gnorm > eps:
                step = min(1.0, alpha * np.sqrt(k) / gnorm)
                Umat += step * G
                w = np.linalg.norm(Umat, axis=0)
                w[w == 0] = eps
                Umat[:] = (Umat @ np.diag(1.0 / w)).clip(min=eps)
                wt_new *= w
        return U0, U1, wt_new


    @staticmethod
    @numba.njit(parallel=True, fastmath=True, cache=True)
    def apply_grad_numba_parallel(U0, U1, wt, r_idx, c_idx, v_vals, alpha, eps):
        """
        Parallel, race-free kernel:
        - Parallelize over components f (columns) -> disjoint writes, no atomics needed.
        - Data term: accumulates per-column grads from (r_idx, c_idx, v_vals).
        - Model term: weighted Grams via (U*wt)^T (U*wt).
        - Same step scaling, column renorm, and absorption into wt
        """
        k = U0.shape[1]
        n0, n1 = U0.shape[0], U1.shape[0]
        n_obs = len(v_vals)
    
        grad0 = np.zeros((n0, k), dtype=U0.dtype)
        grad1 = np.zeros((n1, k), dtype=U1.dtype)
    
        # ---- data term: safe parallelization over f (columns) ----
        for f in numba.prange(k):
            g0_col = np.zeros(n0, dtype=U0.dtype)
            g1_col = np.zeros(n1, dtype=U1.dtype)
            wf = wt[f]
            for i in range(n_obs):
                r = r_idx[i]
                c = c_idx[i]
                v = v_vals[i]
                g0_col[r] += v * (U1[c, f] * wf)
                g1_col[c] += v * (U0[r, f] * wf)
            # write back disjoint column (race-free)
            for i in range(n0):
                grad0[i, f] = g0_col[i]
            for j in range(n1):
                grad1[j, f] = g1_col[j]
    
        # ---- weighted Grams: (U*wt)^T (U*wt) ----
        U1w = np.empty_like(U1)
        U0w = np.empty_like(U0)
        for j in range(n1):
            for f in range(k):
                U1w[j, f] = U1[j, f] * wt[f]
        for i in range(n0):
            for f in range(k):
                U0w[i, f] = U0[i, f] * wt[f]
    
        G1 = U1w.T @ U1w
        G0 = U0w.T @ U0w
    
        # subtract model term once
        grad0 -= U0 @ G1
        grad1 -= U1 @ G0
    
        # ---- update + renorm + absorb scales into wt ----
        U0_new = U0.copy()
        U1_new = U1.copy()
    
        # global step scaling 
        g0 = 0.0
        for i in range(n0):
            for f in range(k):
                g0 += grad0[i, f] * grad0[i, f]
        g0 = np.sqrt(g0)
        if g0 > eps:
            step0 = alpha * (k ** 0.5) / g0
            if step0 > 1.0: step0 = 1.0
            for i in range(n0):
                for f in range(k):
                    U0_new[i, f] += step0 * grad0[i, f]
    
        g1 = 0.0
        for j in range(n1):
            for f in range(k):
                g1 += grad1[j, f] * grad1[j, f]
        g1 = np.sqrt(g1)
        if g1 > eps:
            step1 = alpha * (k ** 0.5) / g1
            if step1 > 1.0: step1 = 1.0
            for j in range(n1):
                for f in range(k):
                    U1_new[j, f] += step1 * grad1[j, f]
    
        # column norms (Numba-friendly) + clamp + absorb into wt
        u_weights = np.empty(k, dtype=U0.dtype)
        v_weights = np.empty(k, dtype=U0.dtype)
        for f in range(k):
            s0 = 0.0; s1 = 0.0
            for i in range(n0): s0 += U0_new[i, f] * U0_new[i, f]
            for j in range(n1): s1 += U1_new[j, f] * U1_new[j, f]
            n0c = np.sqrt(s0); n1c = np.sqrt(s1)
            if n0c < eps: n0c = 1.0
            if n1c < eps: n1c = 1.0
            u_weights[f] = n0c; v_weights[f] = n1c
    
        for f in range(k):
            inv0 = 1.0 / u_weights[f]; inv1 = 1.0 / v_weights[f]
            for i in range(n0):
                val = U0_new[i, f] * inv0
                U0_new[i, f] = val if val > eps else eps
            for j in range(n1):
                val = U1_new[j, f] * inv1
                U1_new[j, f] = val if val > eps else eps
    
        wt_new = wt.copy()
        for f in range(k):
            wt_new[f] = wt_new[f] * u_weights[f] * v_weights[f]
    
        return U0_new, U1_new, wt_new
    

    @staticmethod
    @numba.njit(parallel=True, fastmath=True, cache=True)
    def apply_grad_numba(U0, U1, wt, r_idx, c_idx, v_vals, alpha, eps):
        """
        Parallel, no-diag version.
        - Data term uses elementwise column scaling (U * wt)
        - Gram term uses (U*wt)^T (U*wt)  (equivalent to D (U^T U) D)
        - Column renorm + absorb scales into wt (as in non-numba)
        """
        k = U0.shape[1]
        n0, n1 = U0.shape[0], U1.shape[0]
    
        # --- Gradients from data term: Xt.dot(U1*wt) and Xt.T.dot(U0*wt)
        grad0 = np.zeros((n0, k), dtype=U0.dtype)
        grad1 = np.zeros((n1, k), dtype=U1.dtype)
    
        # Accumulate sparse products explicitly (r_idx, c_idx, v_vals) ~ COO triples
        # grad0[r, f] += v * (U1[c, f] * wt[f])
        # grad1[c, f] += v * (U0[r, f] * wt[f])
        n_obs = len(v_vals)
        for i in numba.prange(n_obs):
            r = r_idx[i]
            c = c_idx[i]
            v = v_vals[i]
            for f in range(k):
                grad0[r, f] += v * (U1[c, f] * wt[f])
                grad1[c, f] += v * (U0[r, f] * wt[f])
    
        # --- Weighted Gram terms: G1 = (U1*wt)^T (U1*wt), G0 = (U0*wt)^T (U0*wt)
        # Build U*w in-place-friendly temporaries
        U1w = np.empty_like(U1)
        U0w = np.empty_like(U0)
        for i in range(n1):
            for f in range(k):
                U1w[i, f] = U1[i, f] * wt[f]
        for i in range(n0):
            for f in range(k):
                U0w[i, f] = U0[i, f] * wt[f]
    
        # Compute G1 = U1w^T U1w, G0 = U0w^T U0w
        G1 = np.zeros((k, k), dtype=U0.dtype)
        G0 = np.zeros((k, k), dtype=U0.dtype)
        # matmul; Numba will lower this efficiently
        G1 = U1w.T @ U1w
        G0 = U0w.T @ U0w
    
        # Subtract model terms: grad0 -= U0 @ G1 ; grad1 -= U1 @ G0
        grad0 -= U0 @ G1
        grad1 -= U1 @ G0
    
        # --- Updates with global step scaling 
        U0_new = U0.copy()
        U1_new = U1.copy()
    
        # L2 norms of gradients
        g0 = 0.0
        for i in range(n0):
            for f in range(k):
                g0 += grad0[i, f] * grad0[i, f]
        g0 = np.sqrt(g0)
    
        if g0 > eps:
            step0 = alpha * (k ** 0.5) / g0
            if step0 > 1.0:
                step0 = 1.0
            for i in range(n0):
                for f in range(k):
                    U0_new[i, f] += step0 * grad0[i, f]
    
        g1 = 0.0
        for i in range(n1):
            for f in range(k):
                g1 += grad1[i, f] * grad1[i, f]
        g1 = np.sqrt(g1)
    
        if g1 > eps:
            step1 = alpha * (k ** 0.5) / g1
            if step1 > 1.0:
                step1 = 1.0
            for i in range(n1):
                for f in range(k):
                    U1_new[i, f] += step1 * grad1[i, f]
    
        # --- Column normalization + positivity floor; absorb scales into wt
        u_weights = np.empty(k, dtype=U0.dtype)
        v_weights = np.empty(k, dtype=U0.dtype)
    
        # compute column norms
        for f in range(k):
            s0 = 0.0
            s1 = 0.0
            for i in range(n0):
                s0 += U0_new[i, f] * U0_new[i, f]
            for i in range(n1):
                s1 += U1_new[i, f] * U1_new[i, f]
            nrm0 = np.sqrt(s0)
            nrm1 = np.sqrt(s1)
            if nrm0 < eps:
                nrm0 = 1.0
            if nrm1 < eps:
                nrm1 = 1.0
            u_weights[f] = nrm0
            v_weights[f] = nrm1
    
        # normalize columns and clamp to eps
        for f in range(k):
            inv0 = 1.0 / u_weights[f]
            inv1 = 1.0 / v_weights[f]
            for i in range(n0):
                val = U0_new[i, f] * inv0
                if val < eps:
                    val = eps
                U0_new[i, f] = val
            for i in range(n1):
                val = U1_new[i, f] * inv1
                if val < eps:
                    val = eps
                U1_new[i, f] = val
    
        # Update W: multiply by the column norms we just divided out
        wt_new = wt.copy()
        for f in range(k):
            wt_new[f] = wt_new[f] * u_weights[f] * v_weights[f]
    
        return U0_new, U1_new, wt_new


    def forecast(self, ridx, current_time, steps_ahead=1):
        U, V = self.U
        future_t = current_time + steps_ahead
        wt = self.W[ridx, future_t - self.s]
        # Avoid forming diag(wt)
        # Equivalent to U @ diag(wt) @ V.T, but faster and allocation-free.
        return (U * wt) @ V.T

    def make_window(self, t: int, s: int) -> List[Tuple[int, List[Tuple[int,int,float]]]]:
        return [(tt, self.triplet_dict.get(tt, [])) for tt in range(t - s + 1, t + 1)]

    def save_results(self, forecasts, rmse_total, rmse_zeros, rmse_nonzeros):
        print("\n--- Final Results")
        print("Average RMSE (total):", np.mean(rmse_total))
        print("Average RMSE (zeros):", np.mean(rmse_zeros))
        print("Average RMSE (nonzeros):", np.mean(rmse_nonzeros))
        
        out = Path(self.output_dir)
        np.save(out / "ssmf_forecasts.npy", np.array(forecasts))
        np.savetxt(out / "rmse_total_ssmf.txt",   np.array(rmse_total))
        np.savetxt(out / "rmse_zeros_ssmf.txt",  np.array(rmse_zeros))
        np.savetxt(out / "rmse_nonzeros_ssmf.txt",np.array(rmse_nonzeros))

        np.save(out / 'U.npy', self.U[0])
        np.save(out / 'V.npy', self.U[1])
        np.save(out / 'W.npy', self.W)
        np.savetxt(out / 'R.txt', self.R, fmt='%d')

        # Conformal diagnostics (if collected)
        # Dual-class conformal diagnostics
        if hasattr(self, "pi_widths_nz") and len(self.pi_widths_nz) > 0:
            np.savetxt(out / "conformal_widths_nz.txt",    np.asarray(self.pi_widths_nz, float))
            np.savetxt(out / "conformal_coverages_nz.txt", np.asarray(self.pi_coverages_nz, float))
        if hasattr(self, "pi_widths_zero") and len(self.pi_widths_zero) > 0:
            np.savetxt(out / "conformal_widths_zero.txt",    np.asarray(self.pi_widths_zero, float))
            np.savetxt(out / "conformal_coverages_zero.txt", np.asarray(self.pi_coverages_zero, float))


        print(f"All results saved to {self.output_dir}")




    # --- Conformal params (feel free to tune defaults) ---
    #conformal_alpha: float = 0.10         # 90% PI
    #conformal_buffer_maxlen: int = 100_000  # # of abs residuals to keep

    def _init_conformal(self):
        from collections import deque
        # rolling buffers of *normalized* absolute residuals
        self._conf_abs_nz   = deque(maxlen=self.conformal_buffer_maxlen)
        self._conf_abs_zero = deque(maxlen=self.conformal_buffer_maxlen)
    
        # logs
        self.pi_widths_nz = []
        self.pi_coverages_nz = []
        self.pi_widths_zero = []
        self.pi_coverages_zero = []
    
        # heteroskedastic scaler
        self._hetero_init()


    def _push_abs_residuals_dual_binwise(self, forecast_t, triples, d_shape, zero_samples_per_slice: int):
        d0, d1 = d_shape
        nz_pushed = 0
        z_pushed = 0
        zr = None
        zc = None
    
        # ---- nonzeros ----
        if triples:
            r, c, v = zip(*triples)
            r = np.asarray(r, np.int32); c = np.asarray(c, np.int32)
            v = np.asarray(v, float)
            yhat = forecast_t[r, c]
            abs_err = np.abs(v - yhat)
            b = self._bin_index(yhat)
            # push by bin
            for val, bi in zip(abs_err, b):
                self._bins_nz[int(bi)].append(float(val))
            nz_pushed = abs_err.size
    
        # ---- zeros (stratified: top-ŷ + uniform) ----
        m = int(zero_samples_per_slice)
        if m > 0:
            # occupied linear indices from this slice's nonzeros
            occ = set()
            if triples:
                occ = {int(rr) * d1 + int(cc) for rr, cc, _ in triples}
    
            N = d0 * d1
            # candidate pool from complement (unique)
            pool_mult = getattr(self, "zero_pool_mult", 10)
            target_pool = min(pool_mult * m, max(0, N - len(occ)))
            pool = set()
            while len(pool) < target_pool:
                idx = int(np.random.randint(0, N))
                if idx in occ or idx in pool:
                    continue
                pool.add(idx)
    
            if pool:
                pool = np.fromiter(pool, dtype=np.int64)
                pr = (pool // d1).astype(np.int32)
                pc = (pool %  d1).astype(np.int32)
                pyhat = forecast_t[pr, pc]
    
                top_frac = getattr(self, "zero_top_frac", 0.6)
                m_top = int(m * top_frac)
                m_uni = m - m_top
    
                # top by ŷ
                if m_top > 0:
                    top_idx = np.argpartition(pyhat, -m_top)[-m_top:]
                else:
                    top_idx = np.array([], dtype=int)
    
                # uniform from the remainder
                mask = np.ones(pyhat.shape[0], dtype=bool)
                if top_idx.size > 0:
                    mask[top_idx] = False
                rest = np.nonzero(mask)[0]
                if m_uni > 0 and rest.size > 0:
                    uni_idx = np.random.choice(rest, size=min(m_uni, rest.size), replace=False)
                else:
                    uni_idx = np.array([], dtype=int)
    
                chosen = np.concatenate([top_idx, uni_idx])
                if chosen.size > 0:
                    zr = pr[chosen]
                    zc = pc[chosen]
                    yhat0 = forecast_t[zr, zc]  # truth=0
                    abs_err0 = np.abs(yhat0)
                    b0 = self._bin_index(yhat0)
                    for val, bi in zip(abs_err0, b0):
                        self._bins_zero[int(bi)].append(float(val))
                    z_pushed = abs_err0.size
    
        return nz_pushed, z_pushed, zr, zc

    
    
    def _conformal_width_from(self, buf):
        if not buf:
            return 1e-6
        arr = np.fromiter(buf, dtype=float)
        return float(np.quantile(arr, 1.0 - self.conformal_alpha))
    


    def _push_abs_residuals_dual(self, forecast_t, triples, d_shape, zero_samples_per_slice: int):
        """
        Push normalized |residuals| into class-specific buffers:
          nz: |v - yhat| / sigma_nz(yhat)
          z : |0 - yhat| / sigma_z (yhat)
        Also returns sampled zero coords for per-slice zero coverage.
        """
        d0, d1 = d_shape
        nz_pushed = 0
        z_pushed = 0
        zr = None; zc = None
    
        # --- Nonzeros ---
        if triples:
            r, c, v = zip(*triples)
            r = np.asarray(r, np.int32); c = np.asarray(c, np.int32)
            v = np.asarray(v, float)
            yhat = forecast_t[r, c]
            if self.hetero_enabled:
                sigma = self._hetero_sigma(0, yhat)
                abs_norm = np.abs(v - yhat) / sigma
            else:
                abs_norm = np.abs(v - yhat)
            for x in abs_norm:
                self._conf_abs_nz.append(float(x))
            nz_pushed = abs_norm.size
    
            # update a,b for class 0 using x = sqrt(yhat), y = |residual|
            if self.hetero_enabled:
                self._hetero_update(0, np.sqrt(np.clip(yhat, 0.0, None)), np.abs(v - yhat))
    
        # --- Zeros (sampled) ---
        m = int(zero_samples_per_slice)
        if m > 0:
            occ = set()
            if triples:
                occ = {int(rr)*d1 + int(cc) for rr, cc, _ in triples}
            N = d0 * d1
            chosen = set()
            while len(chosen) < m and len(chosen) < (N - len(occ)):
                idx = int(np.random.randint(0, N))
                if idx in occ or idx in chosen:
                    continue
                chosen.add(idx)
    
            if chosen:
                chosen = np.fromiter(chosen, dtype=np.int64)
                zr = (chosen // d1).astype(np.int32)
                zc = (chosen %  d1).astype(np.int32)
                yhat0 = forecast_t[zr, zc]  # truth=0
                if self.hetero_enabled:
                    sigma0 = self._hetero_sigma(1, yhat0)
                    abs_norm0 = np.abs(yhat0) / sigma0
                else:
                    abs_norm0 = np.abs(yhat0)
                for x in abs_norm0:
                    self._conf_abs_zero.append(float(x))
                z_pushed = abs_norm0.size
    
                # update a,b for class 1
                if self.hetero_enabled:
                    self._hetero_update(1, np.sqrt(np.clip(yhat0, 0.0, None)), np.abs(yhat0))
    
        return nz_pushed, z_pushed, zr, zc



    def _conformal_width(self):
        """Return q_{1-alpha} of |residuals|; fallback to small epsilon if empty."""
        if len(self._conformal_abs_resid) == 0:
            return 1e-6
        q = np.quantile(np.fromiter(self._conformal_abs_resid, dtype=float),
                        1.0 - self.conformal_alpha)
        self._conformal_ready = True
        return float(q)

    def _conformal_interval(self, forecast_t, width):
        """Broadcast scalar width around matrix forecast."""
        lo = forecast_t - width
        up = forecast_t + width
        # optional: clamp to nonnegative if counts
        np.maximum(lo, 0.0, out=lo)
        return lo, up

    # --- Heteroskedastic scaler state (per class: 0=nonzero, 1=zero) ---
    def _hetero_init(self):
        # normal-equation sufficient stats for y = a + b x  (x = sqrt(yhat), y = |resid|)
        # For each class keep [n, sum_x, sum_xx, sum_y, sum_xy]
        self._hs_stats = {
            0: np.zeros(5, dtype=float),  # nonzeros
            1: np.zeros(5, dtype=float),  # zeros
        }
        # parameters a,b per class
        self._hs_ab = {
            0: np.array([1e-3, 0.0], dtype=float),  # start tiny scale
            1: np.array([1e-6, 0.0], dtype=float),
        }
        self.hetero_enabled = True            # simple on/off knob
        self.hetero_eps = 1e-6                # floor for sigma
        self.hetero_ridge = 1e-8              # tiny ridge for stability
    
    def _hetero_update(self, cls: int, x: np.ndarray, y: np.ndarray):
        """Online WLS (really OLS here) update for a,b using normal eq sums."""
        if x.size == 0: return
        s = self._hs_stats[cls]
        # accumulate
        s[0] += x.size
        s[1] += np.sum(x)
        s[2] += np.sum(x * x)
        s[3] += np.sum(y)
        s[4] += np.sum(x * y)
        self._hs_stats[cls] = s
    
        # solve [ [n, sum_x], [sum_x, sum_xx + λ] ] [a,b] = [sum_y, sum_xy]
        n, sx, sxx, sy, sxy = s
        A00 = n
        A01 = sx
        A11 = sxx + self.hetero_ridge
        det = A00 * A11 - A01 * A01
        if det <= 0:
            # fallback: keep previous a,b
            return
        a = ( A11 * sy - A01 * sxy) / det
        b = (-A01 * sy + A00 * sxy) / det
        # keep nonnegative and reasonable
        if np.isfinite(a) and np.isfinite(b):
            self._hs_ab[cls] = np.array([max(a, 0.0), max(b, 0.0)], dtype=float)
    
    def _hetero_sigma(self, cls: int, yhat: np.ndarray):
        """Return per-entry sigma = a + b*sqrt(yhat), floored at hetero_eps."""
        a, b = self._hs_ab[cls]
        # note: sqrt(ŷ) is well-defined for counts; clamp negatives to 0 just in case
        s = a + b * np.sqrt(np.clip(yhat, 0.0, None))
        return np.maximum(s, self.hetero_eps)


    def _binconformal_init(self):
        from collections import deque
        # edges on sqrt(yhat). Example: 0,1,3,10,+inf (counts)
        self.bin_edges = np.array([0.0, 1.0, 3.0, 10.0, np.inf], dtype=float)
        B = len(self.bin_edges) - 1
        self._bins_nz   = [deque(maxlen=self.conformal_buffer_maxlen // B) for _ in range(B)]
        self._bins_zero = [deque(maxlen=self.conformal_buffer_maxlen // B) for _ in range(B)]
        # logs
        self.pi_widths_nz = []
        self.pi_coverages_nz = []
        self.pi_widths_zero = []
        self.pi_coverages_zero = []
    
    def _bin_index(self, yhat):
        # yhat: np.ndarray of nonnegative predictions
        x = np.sqrt(np.clip(yhat, 0.0, None))
        # digitize returns indices in 1..B, shift to 0..B-1 and clamp
        b = np.digitize(x, self.bin_edges[1:-1], right=True)
        return np.clip(b, 0, len(self.bin_edges) - 2)
    
    def _bin_quantiles(self, bins_deques):
        # return an array q per bin; fallback to small epsilon if empty
        qs = np.zeros(len(bins_deques), dtype=float)
        for i, dq in enumerate(bins_deques):
            if dq:
                arr = np.fromiter(dq, dtype=float)
                qs[i] = np.quantile(arr, 1.0 - self.conformal_alpha)
            else:
                qs[i] = 1e-6
        return qs

def _sample_zero_indices(self, forecast_t, triples, d_shape, m, top_frac=0.5, pool_mult=10):
    """
    Return ~m zero coords (zr, zc), half from the largest unobserved ŷ (stratified),
    half uniform. No dense materialization.
    """
    d0, d1 = d_shape
    if m <= 0:
        return None, None

    # occupied linear indices from nonzeros in this slice
    occ = set()
    if triples:
        occ = {int(rr)*d1 + int(cc) for rr, cc, _ in triples}

    N = d0 * d1
    # sample a *candidate pool* uniformly from complement
    target_pool = min(pool_mult * m, max(0, N - len(occ)))
    pool = set()
    while len(pool) < target_pool:
        idx = int(np.random.randint(0, N))
        if idx in occ or idx in pool:
            continue
        pool.add(idx)
    if not pool:
        return None, None

    pool = np.fromiter(pool, dtype=np.int64)
    pr = (pool // d1).astype(np.int32)
    pc = (pool %  d1).astype(np.int32)
    pyhat = forecast_t[pr, pc]

    m_top = int(m * top_frac)
    m_uni = m - m_top

    # top by ŷ
    if m_top > 0:
        top_idx = np.argpartition(pyhat, -m_top)[-m_top:]
    else:
        top_idx = np.array([], dtype=int)

    # uniform from the rest
    mask = np.ones(pyhat.shape[0], dtype=bool)
    mask[top_idx] = False
    rest = np.nonzero(mask)[0]
    if m_uni > 0 and rest.size > 0:
        uni_idx = np.random.choice(rest, size=min(m_uni, rest.size), replace=False)
    else:
        uni_idx = np.array([], dtype=int)

    chosen = np.concatenate([top_idx, uni_idx])
    if chosen.size == 0:
        return None, None

    return pr[chosen], pc[chosen]


class SSMFStateSnapshot:
    def __init__(self, U, V, regime_id, W_view, period):
        self.U = U.copy()
        self.V = V.copy()
        self.regime_id = int(regime_id)
        self.W_view = W_view          # shape: (g, time, k)
        self.period = int(period)

    @staticmethod
    def _seasonal_slot(current_t: int, target_t: int, s: int) -> int:
        return current_t - ((current_t - target_t) % s)

    def forecast_full(self, current_t: int, target_t: int):
        slot = self._seasonal_slot(current_t, target_t, self.period)
        w = self.W_view[self.regime_id, slot]       # (k,)
        return (self.U * w) @ self.V.T

    def forecast_rc(self, r_idx: np.ndarray, c_idx: np.ndarray,
                    current_t: int, target_t: int) -> np.ndarray:
        slot = self._seasonal_slot(current_t, target_t, self.period)
        w = self.W_view[self.regime_id, slot]       # (k,)
        # μ[r,c] = <U[r,:] * w, V[c,:]>
        return np.einsum('rk,k,ck->r', self.U[r_idx, :], w, self.V[c_idx, :])



# Main()
def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("parquet", help="Parquet file with PU_idx/DO_idx/t_idx/trip_count")
    pa.add_argument("--output_dir", default="out_stream")
    pa.add_argument("--periodicity", type=int, default=24)
    pa.add_argument("--n_components", type=int, default=10)
    pa.add_argument("--max_regimes", type=int, default=50)
    pa.add_argument("--max_iter", type=int, default=1)
    pa.add_argument("--learning_rate", type=float, default=0.2)
    pa.add_argument("--penalty", type=float, default=0.05)
    pa.add_argument("--float_cost", type=int, default=32)
    pa.add_argument("--update_freq", type=int, default=1)
    pa.add_argument("--init_cycles", type=int, default=3)
    pa.add_argument("--conf_alpha", type=float, default=0.10,
                    help="Conformal alpha (e.g., 0.10 for 90% PI)")
    pa.add_argument("--conf_buf", type=int, default=100_000,
                    help="Max calibration buffer size (# abs residuals kept)")
    pa.add_argument("--zero_samples", type=int, default=2000,
                     help="Number of zero entries to sample per slice for conformal")

    pa.add_argument("--unc_mode", choices=["none","homo","binwise","hetero"],
                    default="binwise", help="Uncertainty strategy")
    pa.add_argument("--bin_edges", type=str, default="",
                    help="Comma-separated edges for binwise (on counts); e.g. 0,1,2,3,5,8,13,21,34,55,89")
    pa.add_argument("--bin_Nmin_zero", type=int, default=2000)
    pa.add_argument("--zero_top_frac", type=float, default=0.6)
    pa.add_argument("--zero_pool_mult", type=int, default=10)
    # (hetero-only) optional knobs
    pa.add_argument("--ema_gamma", type=float, default=0.05)
    pa.add_argument("--ridge", type=float, default=1e-6)
    pa.add_argument("--min_sigma", type=float, default=1e-6)
    pa.add_argument("--max_sigma", type=float, default=-1.0, help="<=0 means None")

    args = pa.parse_args()

    edges = None
    if args.unc_mode == "binwise":
        if args.bin_edges.strip():
            edges = np.array([float(x) for x in args.bin_edges.split(",")], dtype=float)
            edges.sort()
            if not np.isinf(edges[-1]):
                edges = np.concatenate([edges, [np.inf]])
        else:
            # sensible default if none provided
            edges = np.array([0,1,2,3,5,8,13,21,34,55,89, np.inf], dtype=float)

    
    

    utils.make_directory(args.output_dir)

    df = pd.read_parquet(args.parquet,
                        columns=["PU_idx", "DO_idx", "t_idx", "trip_count"])\
            .astype({"PU_idx":"int32","DO_idx":"int32",
                    "t_idx":"int32","trip_count":"float32"})
    d1 = df.PU_idx.max() + 1
    d2 = df.DO_idx.max() + 1
    T  = df.t_idx.max() + 1
    print(f"Data: d1={d1}, d2={d2}, T={T}")
    
    triplet_dict = defaultdict(list)
    for r in df.itertuples(index=False):
        triplet_dict[r.t_idx].append((r.PU_idx, r.DO_idx, r.trip_count))

    # Build model
    model = SSMF(
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
        # Uncertainty setup
        uncertainty_mode=(None if args.unc_mode=="none" else args.unc_mode),
        zero_samples_per_slice=args.zero_samples,
        zero_top_frac=args.zero_top_frac,
        zero_pool_mult=args.zero_pool_mult,
        bin_edges=edges,
        bin_Nmin_zero=args.bin_Nmin_zero,
        ema_gamma=args.ema_gamma,
        ridge=args.ridge,
        min_sigma=args.min_sigma,
        max_sigma=(None if args.max_sigma <= 0 else args.max_sigma),
    )

    model.output_dir = args.output_dir
    model.conformal_alpha = args.conf_alpha
    model.conformal_buffer_maxlen = args.conf_buf
    model.zero_samples_per_slice = args.zero_samples
    model.zero_top_frac = 0.8      # fraction of zeros sampled from high-ŷ candidates
    model.zero_pool_mult = 20       # candidate pool size multiplier


    # Streaming fit
    model.fit_stream(T, (d1, d2))              
    print("Saved results to", args.output_dir)

if __name__ == "__main__":
    main()