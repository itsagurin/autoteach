import random
from typing import List, Tuple
from src.tasks.base import BaseTask


class ModularArithmeticTask(BaseTask):
    def __init__(self, p: int = 97, op: str = "+", train_ratio: float = 0.4, seed: int = 42):
        self.p = p
        self.op = op
        self.train_ratio = train_ratio
        self.seed = seed

    @property
    def name(self) -> str:
        op_name = "add" if self.op == "+" else ("mult" if self.op == "*" else "sub")
        return f"modular_{op_name}_p{self.p}"

    def _solve(self, a: int, b: int) -> int:
        if self.op == "+":
            return (a + b) % self.p
        elif self.op == "-":
            return (a - b) % self.p
        elif self.op == "*":
            return (a * b) % self.p
        raise ValueError(f"Unknown operator: {self.op}")

    def get_data(self) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
        rng = random.Random(self.seed)
        all_pairs = [(a, b) for a in range(self.p) for b in range(self.p)]
        rng.shuffle(all_pairs)

        split_idx = int(len(all_pairs) * self.train_ratio)
        train_pairs = all_pairs[:split_idx]
        val_pairs = all_pairs[split_idx:]

        train_data = [(f"{a}{self.op}{b}=", str(self._solve(a, b))) for a, b in train_pairs]
        val_data = [(f"{a}{self.op}{b}=", str(self._solve(a, b))) for a, b in val_pairs]
        return train_data, val_data