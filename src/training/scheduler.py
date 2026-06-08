import math
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import Optimizer

def cosine_lr_schedule(
    optimizer: Optimizer,
    total_epochs: int,
    steps_per_epoch: int,
    warmup_epochs: int = 0,
) -> LambdaLR:
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = total_epochs * steps_per_epoch

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)