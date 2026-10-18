from collections import deque
from typing import Dict, Any, Optional


class GrokDetector:
    """
    Monitors training dynamics to detect:
    1. Memorization vs Generalization gaps
    2. Grokking transitions (delayed generalization)
    3. Stochastic Mini-Batch Equilibrium Traps (frozen loss & stagnant train EM below threshold)
    """

    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.train_loss_history = deque(maxlen=window_size)
        self.val_loss_history = deque(maxlen=window_size)
        self.train_em_history = deque(maxlen=window_size)
        self.val_em_history = deque(maxlen=window_size)
        self.weight_norm_history = deque(maxlen=window_size)

    def update(
            self,
            step: int,
            train_loss: float,
            val_loss: float,
            train_em: float,
            val_em: float,
            weight_norm: float
    ) -> Dict[str, Any]:
        self.train_loss_history.append(train_loss)
        self.val_loss_history.append(val_loss)
        self.train_em_history.append(train_em)
        self.val_em_history.append(val_em)
        self.weight_norm_history.append(weight_norm)

        metrics = {
            "step": step,
            "generalization_gap": round(train_em - val_em, 4),
            "loss_ratio": round(val_loss / max(train_loss, 1e-6), 4),
            "is_equilibrium_trap": False,
            "is_grokked": val_em >= 0.95 and train_em >= 0.98
        }

        if len(self.train_loss_history) >= self.window_size:
            loss_spread = max(self.train_loss_history) - min(self.train_loss_history)
            em_spread = max(self.train_em_history) - min(self.train_em_history)
            mean_train_em = sum(self.train_em_history) / len(self.train_em_history)

            # Stagnant with low variance and suboptimal accuracy = equilibrium trap
            if loss_spread < 0.02 and em_spread < 0.04 and mean_train_em < 0.90:
                metrics["is_equilibrium_trap"] = True
                metrics["trap_severity"] = round(1.0 - mean_train_em, 3)

        return metrics
