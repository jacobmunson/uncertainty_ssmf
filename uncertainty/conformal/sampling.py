# uncertainty/conformal/sampling.py
import numpy as np
from typing import Iterable, Tuple, Optional

Triple = Tuple[int, int, float]
Triples = Iterable[Triple]


def stratified_zero_indices(F, triples: Triples, d_shape, m, top_frac=0.6, pool_mult=10, rng=None):
    """
    Stratified sampling of *zero* cells (unobserved this slice):
      1) Build a small candidate pool from the complement.
      2) Take top-\hat{y} fraction + uniform remainder.

    Deterministic if `rng` is provided. 
    If m<=0 or no zeros available, returns (None, None).
    """
    import numpy as np
    if rng is None:
        rng = np.random.default_rng()

    d0, d1 = d_shape
    if m is None or m <= 0:
        return None, None

    occ = set()
    if triples:
        occ = {int(rr)*d1 + int(cc) for rr, cc, _ in triples}

    N = d0 * d1
    zeros_avail = N - len(occ)
    if zeros_avail <= 0:
        return None, None

    m_eff = min(int(m), zeros_avail)
    target_pool = min(int(pool_mult * m_eff), zeros_avail)
    if target_pool <= 0:
        return None, None

    pool = set()
    while len(pool) < target_pool:
        idx = int(rng.integers(0, N))          # <-- use rng
        if idx in occ or idx in pool:
            continue
        pool.add(idx)

    if not pool:
        return None, None

    pool = np.fromiter(pool, dtype=np.int64)
    pr = (pool // d1).astype(np.int32)
    pc = (pool %  d1).astype(np.int32)
    py = F[pr, pc]

    avail = py.size
    if avail == 0:
        return None, None

    k = min(m_eff, avail)
    m_top = min(int(round(k * float(top_frac))), avail)
    m_uni = max(k - m_top, 0)

    top_idx = np.argpartition(py, -m_top)[-m_top:] if m_top > 0 else np.array([], int)

    if m_uni > 0:
        mask = np.ones(avail, bool)
        if top_idx.size:
            mask[top_idx] = False
        rest = np.nonzero(mask)[0]
        uni_idx = rng.choice(rest, size=min(m_uni, rest.size), replace=False) if rest.size else np.array([], int)
    else:
        uni_idx = np.array([], int)

    chosen = np.concatenate([top_idx, uni_idx])
    if chosen.size == 0:
        return None, None
    return pr[chosen], pc[chosen]