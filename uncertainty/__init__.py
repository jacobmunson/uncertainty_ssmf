# Re-export the conformal factory so both
#   import uncertainty
# and
#   from uncertainty.conformal import make_strategy
# work consistently.
from .conformal import make_strategy  