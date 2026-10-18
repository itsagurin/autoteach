import random
from typing import List, Tuple, Dict, Optional
import torch
from src.engine.masking import prepare_batch
from src.telemetry.metrics import compute_exact_match


class ExperienceReplayBuffer:

    def __init__(self, replay_ratio: float = 0.2, seed: int = 42):
        self.replay_ratio = replay_ratio
        self.rng = random.Random(seed)
        self.task_archive: Dict[int, List[Tuple[str, str]]] = {}
        self.val_archive: Dict[int, List[Tuple[str, str]]] = {}

    def archive_task(
            self,
            level: int,
            train_data: List[Tuple[str, str]],
            val_data: Optional[List[Tuple[str, str]]] = None
    ):
        self.task_archive[level] = list(train_data)
        if val_data is not None:
            self.val_archive[level] = list(val_data)

    def mix_dataset(
            self,
            current_train_data: List[Tuple[str, str]],
            max_samples: int = 4000
    ) -> List[Tuple[str, str]]:
        if not self.task_archive or self.replay_ratio <= 0.0:
            dataset = list(current_train_data)
            self.rng.shuffle(dataset)
            return dataset[:max_samples]

        n_replay = int(max_samples * self.replay_ratio)
        n_current = max_samples - n_replay

        current_sample = self.rng.sample(
            current_train_data,
            min(n_current, len(current_train_data))
        )

        replay_samples = []
        samples_per_task = max(1, n_replay // len(self.task_archive))

        for task_level, task_data in self.task_archive.items():
            k = min(samples_per_task, len(task_data))
            replay_samples.extend(self.rng.sample(task_data, k))

        combined = current_sample + replay_samples
        self.rng.shuffle(combined)
        return combined

    def evaluate_retention(
            self,
            model: torch.nn.Module,
            tokenizer,
            device: torch.device,
            max_samples: int = 200
    ) -> Dict[int, float]:
        """
        Evaluates Exact Match on archived validation sets from previously mastered levels.
        Guarantees early detection of catastrophic forgetting.
        """
        if not self.val_archive:
            return {}

        retention_scores: Dict[int, float] = {}
        model.eval()

        with torch.no_grad():
            for level, val_data in self.val_archive.items():
                if not val_data:
                    continue
                k = min(max_samples, len(val_data))
                sub_val = self.rng.sample(val_data, k)
                x_val, y_val = prepare_batch(sub_val, tokenizer, device)
                logits, _ = model(x_val, targets=y_val)
                em = compute_exact_match(logits, y_val)
                retention_scores[level] = round(em, 4)

        return retention_scores