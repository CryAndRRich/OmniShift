"""Training utilities: epoch loop, LR schedule, sparsity regularization."""

from .train import train_one_epoch, evaluate, EarlyStopping
from .scheduler import cosine_lr_schedule
from .regularize import compute_sparsity_reg

__all__ = [
    "train_one_epoch", "evaluate", "EarlyStopping",
    "cosine_lr_schedule",
    "compute_sparsity_reg",
]
