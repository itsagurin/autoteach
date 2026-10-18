import os
import sys
import time
import yaml
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.tokenizer import MathTokenizer
from src.models.nanogpt import MicroNanoGPT
from src.engine.masking import prepare_batch
from src.tasks import get_task_for_level, ExperienceReplayBuffer
from src.curriculum.fsm_teacher import FSMTeacher, StudentPhase, TeacherAction
from src.telemetry.metrics import compute_weight_norm, compute_exact_match, RollingMetric
from src.telemetry.logger import JSONLinesLogger


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sample_batch(
        data_pairs: list[tuple[str, str]],
        batch_size: int,
        tokenizer: MathTokenizer,
        device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    if batch_size >= len(data_pairs):
        selected = data_pairs
    else:
        indices = torch.randint(0, len(data_pairs), (batch_size,))
        selected = [data_pairs[i] for i in indices]
    return prepare_batch(selected, tokenizer, device)


def main():
    device = select_device()
    print("=" * 70)
    print(f"🚀 RUN CURRICULUM GROKKING LOOP")
    print(f"DEVICE: {device}")
    print("=" * 70)

    with open("configs/student_config.yaml", "r", encoding="utf-8") as f:
        student_cfg = yaml.safe_load(f)
    with open("configs/curriculum_config.yaml", "r", encoding="utf-8") as f:
        curriculum_cfg = yaml.safe_load(f)

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    tokenizer = MathTokenizer()
    model = MicroNanoGPT(
        vocab_size=tokenizer.vocab_size,
        d_model=student_cfg["model"]["d_model"],
        n_layer=student_cfg["model"]["n_layer"],
        n_head=student_cfg["model"]["n_head"],
        block_size=student_cfg["model"]["block_size"]
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(student_cfg["training"]["lr"]),
        betas=tuple(student_cfg["training"]["betas"]),
        weight_decay=float(student_cfg["training"]["weight_decay"])
    )

    max_levels = 4
    teacher = FSMTeacher(curriculum_cfg, initial_level=1, max_levels=max_levels)
    replay_buffer = ExperienceReplayBuffer(replay_ratio=0.2, seed=42)
    logger = JSONLinesLogger(filepath="logs/autonomous_loop.jsonl")
    val_em_smoother = RollingMetric(window_size=5)

    current_task = get_task_for_level(teacher.current_level)
    raw_train_data, raw_val_data = current_task.get_data()
    active_train_data = replay_buffer.mix_dataset(raw_train_data, max_samples=4000)

    x_val, y_val = prepare_batch(raw_val_data, tokenizer, device)

    print(f"\n[Level 1/{max_levels}]: {current_task.name.upper()}")
    print(f"Training examples: {len(active_train_data)} | Validation examples: {len(raw_val_data)}")
    print(
        f"{'Step':<7} | {'Level':<5} | {'Phase':<20} | {'Train EM':<9} | {'Val EM':<9} | {'||W||_2':<7} | {'Action':<15}")
    print("-" * 85)

    total_max_steps = 100_000
    eval_interval = student_cfg["training"]["eval_interval"]
    batch_size = student_cfg["training"]["batch_size"]

    start_time = time.time()

    for step in range(1, total_max_steps + 1):
        model.train()
        optimizer.zero_grad()

        x_train, y_train = sample_batch(active_train_data, batch_size, tokenizer, device)

        logits, loss = model(x_train, targets=y_train)
        loss.backward()
        optimizer.step()

        if step % eval_interval == 0:
            model.eval()
            with torch.no_grad():
                train_em = compute_exact_match(logits, y_train)
                val_logits, val_loss = model(x_val, targets=y_val)
                val_em = compute_exact_match(val_logits, y_val)
                smooth_val_em = val_em_smoother.update(val_em)
                w_norm = compute_weight_norm(model)

            phase, action, meta = teacher.evaluate_step(
                step=step,
                train_loss=loss.item(),
                train_em=train_em,
                val_em=smooth_val_em,
                weight_norm=w_norm,
                optimizer=optimizer
            )

            current_wd = optimizer.param_groups[0]["weight_decay"]
            logger.log(step, {
                "level": teacher.current_level,
                "task": current_task.name,
                "phase": phase.value,
                "action": action.value,
                "train_loss": round(loss.item(), 5),
                "val_loss": round(val_loss.item(), 5),
                "train_em": round(train_em, 4),
                "val_em": round(val_em, 4),
                "smooth_val_em": round(smooth_val_em, 4),
                "weight_norm": round(w_norm, 2),
                "weight_decay": round(current_wd, 4)
            })

            print(
                f"{step:<7} | L{teacher.current_level:<4} | {phase.value:<20} | {train_em:<9.2%} | {val_em:<9.2%} | {w_norm:<7.2f} | {action.value:<15}")


            if action == TeacherAction.BOOST_REGULARIZATION:
                print(
                    f"  └─► [FSM INTERVENTION] Stagnation in weight norms detected. Weight decay increased: {meta['old_wd']:.2f} -> {meta['new_wd']:.2f}")

            elif action == TeacherAction.PROMOTE_NEXT_LEVEL:
                old_level = teacher.current_level - 1
                print("\n" + "=" * 85)
                print(f"🏆 LEVEL {old_level} ({current_task.name.upper()}) SUCCESSFULLY MASTERED AND GENERALIZED!")

                ckpt_path = f"checkpoints/checkpoint_level_{old_level}.pt"
                torch.save({
                    "step": step,
                    "level": old_level,
                    "task": current_task.name,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_em": val_em,
                    "weight_norm": w_norm
                }, ckpt_path)
                print(f"💾 Model weights are saved in {ckpt_path}")

                replay_buffer.archive_task(level=old_level, train_data=raw_train_data)

                current_task = get_task_for_level(teacher.current_level)
                raw_train_data, raw_val_data = current_task.get_data()

                active_train_data = replay_buffer.mix_dataset(raw_train_data, max_samples=4000)
                x_val, y_val = prepare_batch(raw_val_data, tokenizer, device)

                val_em_smoother = RollingMetric(window_size=5)

                print(f"⏩ ADVANCING TO THE NEXT LEVEL {teacher.current_level}/{max_levels}: {current_task.name.upper()}")
                print(
                    f"   New training pool: {len(active_train_data)} examples (including 20% repetitions of past tasks)")
                print(f"   New test: {len(raw_val_data)} examples")
                print("=" * 85 + "\n")

            elif meta.get("status") == "curriculum_mastered":
                total_time = (time.time() - start_time) / 60
                print("\n" + "*" * 85)
                print(f"🎓 (ALL {max_levels} LEVELS) !")
                print(f"TIME: {total_time:.2f} MIN | STEP: {step}")
                print(f"Final checkpoint saved: checkpoints/final_model_mastered.pt")
                print("*" * 85)

                torch.save(model.state_dict(), "checkpoints/final_model_mastered.pt")
                break


if __name__ == "__main__":
    main()