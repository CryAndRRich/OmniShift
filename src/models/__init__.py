"""Model definitions for OmniShift phases 1–5."""

from .resnet20 import ResNet20, build_model as build_baseline
from .resnet20_potbn import ResNet20PoTBN, build_model as build_potbn
from .resnet20_sparse import ResNet20Sparse, build_model as build_sparse
from .resnet20_full import ResNet20SparsePoTBN, build_model, parse_config_name

__all__ = [
    "ResNet20",
    "ResNet20PoTBN",
    "ResNet20Sparse",
    "ResNet20SparsePoTBN",
    "build_model",
    "parse_config_name",
    "build_baseline",
    "build_potbn",
    "build_sparse",
]
