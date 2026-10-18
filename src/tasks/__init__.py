from src.tasks.base import BaseTask
from src.tasks.modular import ModularArithmeticTask
from src.tasks.multi_digit import MultiDigitAdditionTask
from src.tasks.scratchpad import ScratchpadArithmeticTask
from src.tasks.replay_buffer import ExperienceReplayBuffer


def get_task_for_level(level: int) -> BaseTask:
    """
    Problem Factory for a Complexity Ladder (Curriculum Ladder):
    Level 1: Modular addition (+) p=97
    Level 2: Modular multiplication (*) p=97
    Level 3: Multi-digit addition with carrying (2–3 digits)
    Level 4: Complex arithmetic using a scratchpad (Chain-of-Thought)
    """
    curriculum = {
        1: ModularArithmeticTask(p=97, op="+", train_ratio=0.4, seed=42),
        2: ModularArithmeticTask(p=97, op="*", train_ratio=0.25, seed=42),
        3: MultiDigitAdditionTask(num_samples=6000, train_ratio=0.5, seed=42),
        4: ScratchpadArithmeticTask(num_samples=5000, train_ratio=0.5, seed=42),
    }
    if level not in curriculum:
        raise ValueError(f"Unknown curriculum level: {level}")
    return curriculum[level]