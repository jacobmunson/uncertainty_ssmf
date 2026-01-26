# uncertainty/conformal/quantile.py
import numpy as np
from collections import deque

def conformal_q(buf, alpha: float, eps: float = 1e-6) -> float:
    """
    Finite-sample conformal quantile:
        q = ceil((m+1)(1-alpha))-th order statistic
    """
    if not buf:
        return eps

    a = np.fromiter(buf, float)
    m = a.size
    k = int(np.ceil((m + 1) * (1.0 - alpha)))
    k = np.clip(k, 1, m)
    return float(np.partition(a, k - 1)[k - 1])