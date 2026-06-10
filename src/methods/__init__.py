from .fp32 import make_factories as _fp32
from .deepshift import make_factories as _deepshift
from .apot import make_factories as _apot
from .denseshift import make_factories as _denseshift
from .s3shift import make_factories as _s3shift
from .fogzo import make_factories as _fogzo
from .aptq_ternary import make_factories as _aptq
from .omnishift import make_factories as _omnishift

_REGISTRY = {
    "fp32": _fp32,
    "deepshift": _deepshift,
    "apot": _apot,
    "denseshift": _denseshift,
    "s3shift": _s3shift,
    "fogzo": _fogzo,
    "aptq": _aptq,
    "omnishift": _omnishift
}

METHODS = list(_REGISTRY)

def get_factories(method: str, **opts):
    if method not in _REGISTRY:
        raise ValueError(
            f"Unknown method {method!r}. Choose from: {METHODS}"
        )
    return _REGISTRY[method](**opts)