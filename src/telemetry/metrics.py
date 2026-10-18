import torch
from collections import deque


def compute_weight_norm(model: torch.nn.Module) -> float:
    total_norm_sq = 0.0
    for param in model.parameters():
        if param.requires_grad:
            total_norm_sq += param.data.norm(2).item() ** 2
    return total_norm_sq ** 0.5


def compute_token_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    mask = targets != -100
    if mask.sum() == 0:
        return 0.0
    correct = (preds[mask] == targets[mask]).sum().item()
    return correct / mask.sum().item()


def compute_exact_match(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    mask = targets != -100

    token_matches = (preds == targets) | (~mask)

    sample_correct = token_matches.all(dim=-1)
    return sample_correct.float().mean().item()


class RollingMetric:
    def __init__(self, window_size: int = 5):
        self.window = deque(maxlen=window_size)

    def update(self, value: float) -> float:
        self.window.append(value)
        return sum(self.window) / len(self.window)

    @property
    def value(self) -> float:
        return sum(self.window) / len(self.window) if self.window else 0.0