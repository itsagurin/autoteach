import random
from typing import List, Tuple, Dict


class ExperienceReplayBuffer:

    def __init__(self, replay_ratio: float = 0.2, seed: int = 42):
        self.replay_ratio = replay_ratio
        self.rng = random.Random(seed)
        self.task_archive: Dict[int, List[Tuple[str, str]]] = {}

    def archive_task(self, level: int, train_data: List[Tuple[str, str]]):
        self.task_archive[level] = list(train_data)

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