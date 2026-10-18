import random
from typing import List, Tuple, Optional
import torch
import torch.nn as nn
from src.models.tokenizer import MathTokenizer
from src.engine.masking import prepare_batch
from src.telemetry.metrics import compute_exact_match, compute_weight_norm


class EpochBatchSampler:
    """
    Epoch-based sampler that shuffles data and samples without replacement.
    Prevents sample starvation and high-variance noise caused by sampling with replacement.
    """

    def __init__(self, data: List[Tuple[str, str]], batch_size: int, seed: int = 42):
        self.data = data
        self.batch_size = batch_size
        self.rng = random.Random(seed)
        self.indices = list(range(len(self.data)))
        self.current_idx = 0
        self.rng.shuffle(self.indices)

    def update_data(self, data: List[Tuple[str, str]]):
        self.data = data
        self.indices = list(range(len(self.data)))
        self.current_idx = 0
        self.rng.shuffle(self.indices)

    def next_batch(self) -> List[Tuple[str, str]]:
        if self.batch_size >= len(self.data) or self.batch_size <= 0:
            return self.data

        if self.current_idx + self.batch_size > len(self.indices):
            self.rng.shuffle(self.indices)
            self.current_idx = 0

        batch_indices = self.indices[self.current_idx: self.current_idx + self.batch_size]
        self.current_idx += self.batch_size
        return [self.data[i] for i in batch_indices]


class StudentTrainer:
    """
    Unified trainer for MicroNanoGPT across curriculum levels.
    Supports full-batch mode, gradient accumulation, and adaptive escape from equilibrium traps.
    """

    def __init__(
            self,
            model: nn.Module,
            optimizer: torch.optim.Optimizer,
            tokenizer: MathTokenizer,
            device: torch.device,
            batch_size: int = 512,
            gradient_accumulation_steps: int = 1,
            full_batch_mode: bool = False
    ):
        self.model = model
        self.optimizer = optimizer
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = batch_size
        self.gradient_accumulation_steps = max(1, gradient_accumulation_steps)
        self.full_batch_mode = full_batch_mode

        self.sampler: Optional[EpochBatchSampler] = None
        self.cached_train_tensors: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self.cached_val_tensors: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    def set_datasets(
            self,
            train_data: List[Tuple[str, str]],
            val_data: List[Tuple[str, str]],
            is_modular_task: bool = False,
            force_full_batch: bool = False
    ):
        """
        Configures active datasets and caches tensors for fast deterministic evaluation.
        """
        self.sampler = EpochBatchSampler(train_data, self.batch_size)

        if force_full_batch or self.full_batch_mode or self.batch_size <= 0 or self.batch_size >= len(train_data):
            self.active_full_batch = True
        else:
            self.active_full_batch = False

        self.cached_train_tensors = prepare_batch(train_data, self.tokenizer, self.device)
        self.cached_val_tensors = prepare_batch(val_data, self.tokenizer, self.device)

    def train_step(self) -> Tuple[float, float]:
        """
        Executes one forward/backward optimization step with gradient clipping.
        Returns (train_loss, train_exact_match).
        """
        self.model.train()
        self.optimizer.zero_grad()

        if self.active_full_batch and self.cached_train_tensors is not None:
            x_train, y_train = self.cached_train_tensors
            logits, loss = self.model(x_train, targets=y_train)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            train_loss = loss.item()
            with torch.no_grad():
                train_em = compute_exact_match(logits, y_train)
            return train_loss, train_em

        # Mini-batch mode with gradient accumulation support
        total_loss = 0.0
        last_logits = None
        last_targets = None

        for _ in range(self.gradient_accumulation_steps):
            batch_pairs = self.sampler.next_batch()
            x_b, y_b = prepare_batch(batch_pairs, self.tokenizer, self.device)
            logits, loss = self.model(x_b, targets=y_b)

            scaled_loss = loss / self.gradient_accumulation_steps
            scaled_loss.backward()

            total_loss += loss.item()
            last_logits = logits
            last_targets = y_b

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        avg_loss = total_loss / self.gradient_accumulation_steps

        with torch.no_grad():
            batch_em = compute_exact_match(last_logits, last_targets)

        return avg_loss, batch_em

    def evaluate(self) -> Tuple[float, float, float, float, float]:
        """
        Runs deterministic evaluation on full validation set and full train evaluation set.
        Returns: (val_loss, val_em, train_eval_loss, train_eval_em, weight_norm).
        """
        self.model.eval()
        with torch.no_grad():
            x_val, y_val = self.cached_val_tensors
            val_logits, val_loss = self.model(x_val, targets=y_val)
            val_em = compute_exact_match(val_logits, y_val)

            x_train, y_train = self.cached_train_tensors
            train_logits, train_loss = self.model(x_train, targets=y_train)
            train_eval_em = compute_exact_match(train_logits, y_train)

            w_norm = compute_weight_norm(self.model)

        return val_loss.item(), val_em, train_loss.item(), train_eval_em, w_norm

    def escape_equilibrium_trap(self):
        """
        Intervention triggered when the student is trapped in an equilibrium trap:
        Switches to full-batch or doubles gradient accumulation steps to reduce gradient variance.
        """
        if not self.active_full_batch:
            self.active_full_batch = True
            return "switched_to_full_batch"
        else:
            self.gradient_accumulation_steps *= 2
            return f"increased_gradient_accumulation_to_{self.gradient_accumulation_steps}"
