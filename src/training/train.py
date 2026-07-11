import torch
import torch.nn as nn

from .regularize import compute_sparsity_reg

def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer,
    scheduler,
    device,
    scaler=None,
    clip_grad: float = 0.0,
    sparsity_lambda: float = 0.0,
) -> tuple[float, float]:
    model.train()
    crit = nn.CrossEntropyLoss()
    loss_sum = correct = total = 0
    use_amp = scaler is not None

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.cuda.amp.autocast(dtype=torch.float16):
                logits = model(imgs)
                loss_ce = crit(logits, labels)
                loss_reg = compute_sparsity_reg(model, sparsity_lambda)
                loss = loss_ce + loss_reg
            scaler.scale(loss).backward()
            if clip_grad > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(imgs)
            loss_ce = crit(logits, labels)
            loss_reg = compute_sparsity_reg(model, sparsity_lambda)
            loss = loss_ce + loss_reg
            loss.backward()
            if clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()

        scheduler.step()

        loss_sum += loss_ce.item() * imgs.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += imgs.size(0)

    return loss_sum / total, correct / total

@torch.no_grad()
def reestimate_bn(model: nn.Module, loader, device, n_batches: int = 50) -> None:
    """Re-estimate BN running statistics with the current (quantized) weights
    (Nagel et al., ICML 2022). Uses momentum = 1/(i+1) so the result is the
    cumulative average over the batches seen. Runs under no_grad, so
    hysteresis/mask/freeze buffers in quantized modules do not advance."""
    bns = [m for m in model.modules()
           if hasattr(m, "running_mean") and hasattr(m, "momentum")]
    if not bns:
        return
    saved = [m.momentum for m in bns]
    was_training = model.training
    model.train()
    for i, (imgs, _) in enumerate(loader):
        if i >= n_batches:
            break
        for m in bns:
            m.momentum = 1.0 / (i + 1)
        model(imgs.to(device, non_blocking=True))
    for m, mom in zip(bns, saved):
        m.momentum = mom
    model.train(was_training)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    device,
    use_amp: bool = False,
) -> tuple[float, float]:
    model.eval()
    crit = nn.CrossEntropyLoss()
    loss_sum = correct = total = 0

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if use_amp:
            with torch.cuda.amp.autocast(dtype=torch.float16):
                logits = model(imgs)
                loss = crit(logits, labels)
        else:
            logits = model(imgs)
            loss = crit(logits, labels)
        loss_sum += loss.item() * imgs.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += imgs.size(0)

    return loss_sum / total, correct / total

class EarlyStopping:
    def __init__(self, patience: int = 25, min_delta: float = 1e-4,
                 min_epochs: int = 50):
        self.patience = patience
        self.min_delta = min_delta
        self.min_epochs = min_epochs
        self.best = -float('inf')
        self.counter = 0
        self.should_stop = False

    def step(self, metric: float, epoch: int) -> bool:
        improved = metric > self.best + self.min_delta
        if improved:
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1
        if epoch + 1 >= self.min_epochs and self.counter >= self.patience:
            self.should_stop = True
        return improved