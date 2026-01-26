uncertainty/conformal/
    
    __init__.py          # make_strategy(...) factory

    base.py              # Strategy protocol (start/step/save)
    
    sampling.py          # stratified_zero_indices(...)
    
    homo.py              # homoskedastic dual-buffers
    
    binwise.py           # binwise-by-magnitude dual-buffers 
    
    hetero.py            # a+b*sqrt(yhat) scaler dual-buffers
