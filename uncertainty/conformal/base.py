# uncertainty/conformal/base.py
from typing import Protocol, Tuple, Iterable
from pathlib import Path
import numpy as np

Triple = Tuple[int, int, float]
Triples = Iterable[Triple]

class ConformalStrategy(Protocol):
    def start(self, *, alpha: float, buffer_maxlen: int, model=None) -> None: ...
    def step(self, forecast_t: np.ndarray, next_triples: Triples,
             d_shape: Tuple[int,int]) -> Tuple[float, float, float, float]:
        """Return (avg_width_nz, cov_nz, avg_width_zero, cov_zero) for this slice."""
        ...
    def save(self, out_dir: Path) -> None: ...