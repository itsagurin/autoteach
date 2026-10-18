import random
from typing import List, Tuple
from src.tasks.base import BaseTask


class ScratchpadArithmeticTask(BaseTask):
    def __init__(self, num_samples: int = 6000, train_ratio: float = 0.5, seed: int = 42):
        self.num_samples = num_samples
        self.train_ratio = train_ratio
        self.seed = seed

    @property
    def name(self) -> str:
        return "scratchpad_composite_arithmetic"

    def get_data(self) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
        rng = random.Random(self.seed)
        samples = []
        seen = set()

        while len(samples) < self.num_samples:
            a = rng.randint(2, 20)
            b = rng.randint(2, 20)
            c = rng.randint(1, 99)
            if (a, b, c) not in seen:
                seen.add((a, b, c))
                samples.append((a, b, c))

        rng.shuffle(samples)
        split_idx = int(len(samples) * self.train_ratio)

        train_triplets = samples[:split_idx]
        val_triplets = samples[split_idx:]

        def format_sample(a: int, b: int, c: int) -> Tuple[str, str]:
            mult_res = a * b
            final_res = mult_res + c
            prompt = f"{a}*{b}+{c}="
            answer = f"[{a}*{b}={mult_res}][{mult_res}+{c}={final_res}]={final_res}"
            return prompt, answer

        train_data = [format_sample(a, b, c) for a, b, c in train_triplets]
        val_data = [format_sample(a, b, c) for a, b, c in val_triplets]
        return train_data, val_data