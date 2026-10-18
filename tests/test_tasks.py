import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.tasks import get_task_for_level, ExperienceReplayBuffer


def test_task_data_integrity():
    for level in range(1, 5):
        task = get_task_for_level(level)
        train_data, val_data = task.get_data()

        train_prompts = {p for p, _ in train_data}
        val_prompts = {p for p, _ in val_data}

        leakage = train_prompts.intersection(val_prompts)
        assert len(leakage) == 0, f"Data leak detected at the level of {level}: {leakage}"
        assert len(train_data) > 0 and len(val_data) > 0
        print(f"[OK] Level {level} ({task.name}): Train={len(train_data)}, Val={len(val_data)}, Data Leakage=0")


def test_replay_buffer_proportions():
    buffer = ExperienceReplayBuffer(replay_ratio=0.2, seed=42)

    task1_train = [(f"t1_{i}=", str(i)) for i in range(1000)]
    task2_train = [(f"t2_{i}=", str(i)) for i in range(1000)]
    buffer.archive_task(level=1, train_data=task1_train)
    buffer.archive_task(level=2, train_data=task2_train)

    task3_train = [(f"t3_{i}=", str(i)) for i in range(1000)]
    mixed_batch = buffer.mix_dataset(task3_train, max_samples=100)

    assert len(mixed_batch) == 100

    current_count = sum(1 for p, _ in mixed_batch if p.startswith("t3_"))
    replay_count = sum(1 for p, _ in mixed_batch if p.startswith("t1_") or p.startswith("t2_"))

    print(f"[OK] Replay Buffer: Current samples = {current_count}, Replay samples = {replay_count}")
    assert 75 <= current_count <= 85, "The proportion of current data is off (expected ~80%)"
    assert 15 <= replay_count <= 25, "The proportion of archival data is off (expected ~20%)"


if __name__ == "__main__":
    test_task_data_integrity()
    test_replay_buffer_proportions()
    print("\nAll tests are passed")