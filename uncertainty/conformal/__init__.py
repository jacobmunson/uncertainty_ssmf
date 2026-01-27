from .homo import HomoStrategy
from .binwise import BinwiseStrategy

try:
    from .hetero import HeteroStrategy
    _HET_AVAILABLE = True
except Exception:
    _HET_AVAILABLE = False

def make_strategy(mode: str, **kwargs):
    m = (mode or "homo").lower()
    if m in ("homo",):
        return HomoStrategy(**kwargs)
    if m in ("binwise", "bin"):
        return BinwiseStrategy(**kwargs)
    if m in ("hetero", "het"):
        if not _HET_AVAILABLE:
            raise RuntimeError("hetero not available")
        return HeteroStrategy(**kwargs)
    if m in ("bestn", "bestnregimes", "best-n", "best_n"):
        return BestNRegimeStrategy(**kwargs)
    if m in ("inject", "injectnoise", "inject-noise", "injectnoiseensemble"):  # <-- add this block
        return InjectNoiseEnsembleStrategy(**kwargs)
    if m in ("kmodels", "k-ensemble", "ensemblek", "ensk"):
        return KModelEnsembleStrategy(**kwargs)
    if m in ("mcdo", "dropout", "mc_dropout"):
        return MCDropoutEnsembleStrategy(**kwargs)
    if m in ("kruns","k-data","kdata","k_boot"):
        return KRunsDataEnsembleStrategy(**kwargs)
    if m in ("factorvar", "factorvars", "factor-var", "factor_var", "fvar"):
        return FactorVarianceStrategy(**kwargs)    
    raise ValueError(f"Unknown uncertainty mode: {mode}")
