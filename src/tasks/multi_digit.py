import random
from typing import List, Tuple
from src.tasks.base import BaseTask


class MultiDigitAdditionTask(BaseTask):

    def __init__(self, num_samples: int = 8000, train_ratio: float = 0.5, seed: int = 42):
        self.num_samples = num_samples
        self.train_ratio = train_ratio
        self.seed = seed

    @property
    def name(self) -> str:
        return "multi_digit_addition"

    def get_data(self) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
        rng = random.Random(self.seed)
        seen_pairs = set()
        samples = []

        while len(samples) < self.num_samples:
            a = rng.randint(10, 999)
            b = rng.randint(10, 999)
            if (a, b) not in seen_pairs:
                seen_pairs.add((a, b))
                samples.append((a, b))

        rng.shuffle(samples)
        split_idx = int(len(samples) * self.train_ratio)

        train_pairs = samples[:split_idx]
        val_pairs = samples[split_idx:]

        train_data = [(f"{a}+{b}=", str(a + b)) for a, b in train_pairs]
        val_data = [(f"{a}+{b}=", str(a + b)) for a, b in val_pairs]
        return train_data, val_data