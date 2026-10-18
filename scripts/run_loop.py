import os
import sys
import time
import yaml
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.tokenizer import MathTokenizer
from src.models.nanogpt import MicroNanoGPT
from src.engine.trainer import StudentTrainer
from src.tasks import get_task_for_level, ExperienceReplayBuffer
from src.curriculum.fsm_teacher import FSMTeacher, StudentPhase, TeacherAction
from src.telemetry.metrics import RollingMetric
from src.telemetry.logger import JSONLinesLogger
from src.telemetry.grok_detector import GrokDetector


import argparse


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    parser = argparse.ArgumentParser(description="Curriculum Grokking Loop")
    parser.add_argument("--resume", action="store_true", help="Resume from latest level checkpoint if available")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to specific checkpoint to resume from")
    args = parser.parse_args()

    device = select_device()
    print("=" * 70)
    print(f"🚀 RUN CURRICULUM GROKKING LOOP (EQUILIBRIUM TRAP RESILIENT)")
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

    base_lr = float(student_cfg["training"]["lr"])
    base_wd = float(student_cfg["training"]["weight_decay"])

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=base_lr,
        betas=tuple(student_cfg["training"]["betas"]),
        weight_decay=base_wd
    )

    max_levels = 4
    start_level = 1
    start_step = 1
    replay_buffer = ExperienceReplayBuffer(replay_ratio=0.2, seed=42)

    if args.checkpoint or args.resume:
        ckpt_path = args.checkpoint
        if not ckpt_path and os.path.exists("checkpoints"):
            available = sorted([f for f in os.listdir("checkpoints") if f.startswith("checkpoint_level_") and f.endswith(".pt")])
            if available:
                ckpt_path = os.path.join("checkpoints", available[-1])
        if ckpt_path and os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict_flexible(ckpt["model_state_dict"])
            start_level = min(ckpt.get("level", 1) + 1, max_levels)
            start_step = ckpt.get("step", 0) + 1
            for lvl in range(1, start_level):
                prev_t = get_task_for_level(lvl)
                prev_train, prev_val = prev_t.get_data()
                replay_buffer.archive_task(level=lvl, train_data=prev_train, val_data=prev_val)
            print(f"🔄 Resumed from '{ckpt_path}' -> Starting at Level {start_level} (Step {start_step})")

    lvl_cfg = curriculum_cfg["curriculum"].get("level_configs", {}).get(start_level, {})
    task_wd = float(lvl_cfg.get("base_weight_decay", base_wd if start_level in [1, 2] else 0.01))
    for g in optimizer.param_groups:
        g["weight_decay"] = task_wd

    teacher = FSMTeacher(curriculum_cfg, initial_level=start_level, max_levels=max_levels)
    teacher.base_weight_decay = task_wd

    logger = JSONLinesLogger(filepath="logs/autonomous_loop.jsonl")
    grok_detector = GrokDetector(window_size=10)
    val_em_smoother = RollingMetric(window_size=5)

    trainer = StudentTrainer(
        model=model,
        optimizer=optimizer,
        tokenizer=tokenizer,
        device=device,
        batch_size=int(student_cfg["training"].get("batch_size", 512)),
        gradient_accumulation_steps=int(student_cfg["training"].get("gradient_accumulation_steps", 1)),
        full_batch_mode=bool(student_cfg["training"].get("full_batch_for_modular", True))
    )

    current_task = get_task_for_level(teacher.current_level)
    raw_train_data, raw_val_data = current_task.get_data()
    active_train_data = replay_buffer.mix_dataset(raw_train_data, max_samples=4000)

    is_modular = "modular" in current_task.name
    is_modular_mult = "modular_mult" in current_task.name
    trainer.set_datasets(active_train_data, raw_val_data, is_modular_task=is_modular, force_full_batch=is_modular_mult)

    print(f"\n[Level {teacher.current_level}/{max_levels}]: {current_task.name.upper()}")
    print(f"Training examples: {len(active_train_data)} | Validation examples: {len(raw_val_data)}")
    print(f"Execution mode: {'FULL-BATCH' if trainer.active_full_batch else f'MINI-BATCH (size={trainer.batch_size})'}")
    print(
        f"{'Step':<7} | {'Level':<5} | {'Phase':<20} | {'Train EM':<9} | {'Val EM':<9} | {'||W||_2':<7} | {'Action':<15}")
    print("-" * 85)

    total_max_steps = int(student_cfg["training"].get("max_steps", 100_000))
    eval_interval = int(student_cfg["training"].get("eval_interval", 200))

    start_time = time.time()

    for step in range(start_step, total_max_steps + 1):
        step_train_loss, step_train_em = trainer.train_step()

        if step % eval_interval == 0:
            val_loss, val_em, train_eval_loss, train_eval_em, w_norm = trainer.evaluate()
            smooth_val_em = val_em_smoother.update(val_em)

            phase, action, meta = teacher.evaluate_step(
                step=step,
                train_loss=train_eval_loss,
                train_em=train_eval_em,
                val_em=max(val_em, smooth_val_em),
                weight_norm=w_norm,
                optimizer=optimizer
            )

            current_wd = optimizer.param_groups[0]["weight_decay"]
            detector_metrics = grok_detector.update(
                step=step,
                train_loss=train_eval_loss,
                val_loss=val_loss,
                train_em=train_eval_em,
                val_em=val_em,
                weight_norm=w_norm
            )

            logger.log(step, {
                "level": teacher.current_level,
                "task": current_task.name,
                "phase": phase.value,
                "action": action.value,
                "train_loss": round(train_eval_loss, 5),
                "val_loss": round(val_loss, 5),
                "train_em": round(train_eval_em, 4),
                "val_em": round(val_em, 4),
                "smooth_val_em": round(smooth_val_em, 4),
                "weight_norm": round(w_norm, 2),
                "weight_decay": round(current_wd, 4),
                "is_equilibrium_trap": detector_metrics.get("is_equilibrium_trap", False)
            })

            print(
                f"{step:<7} | L{teacher.current_level:<4} | {phase.value:<20} | {train_eval_em:<9.2%} | {val_em:<9.2%} | {w_norm:<7.2f} | {action.value:<15}")

            if teacher.current_level > 1 and step % (eval_interval * 2) == 0:
                retention_scores = replay_buffer.evaluate_retention(model, tokenizer, device, max_samples=200)
                if retention_scores:
                    ret_str = " | ".join([f"L{lvl}: {score:.1%}" for lvl, score in sorted(retention_scores.items())])
                    print(f"  └─► [EXPERIENCE REPLAY RETENTION] {ret_str}")

            if action == TeacherAction.BOOST_REGULARIZATION:
                print(
                    f"  └─► [FSM INTERVENTION] Plateau norm stagnation detected. Weight decay increased: {meta.get('old_wd', current_wd):.2f} -> {meta.get('new_wd', current_wd):.2f}")

            elif action == TeacherAction.REDUCE_REGULARIZATION:
                print(
                    f"  └─► [FSM INTERVENTION] Excessive Regularization detected! Regularization reduced: {meta.get('old_wd', current_wd):.2f} -> {meta.get('new_wd', current_wd):.2f}")

            elif action == TeacherAction.ESCAPE_EQUILIBRIUM:
                fix_desc = trainer.escape_equilibrium_trap()
                print(
                    f"  └─► [FSM INTERVENTION] Stochastic Mini-Batch Equilibrium Trap detected! Strategy: {fix_desc}")

            elif action == TeacherAction.ROLLBACK_LR:
                lr_decay = float(curriculum_cfg["curriculum"]["interventions"].get("lr_decay_on_stagnation", 0.8))
                for g in optimizer.param_groups:
                    g["lr"] *= lr_decay
                print(f"  └─► [FSM INTERVENTION] Rolling back learning rate to {optimizer.param_groups[0]['lr']:.6f}")

            elif action == TeacherAction.PROMOTE_NEXT_LEVEL:
                old_level = teacher.current_level - 1
                print("\n" + "=" * 85)
                print(f"🏆 LEVEL {old_level} ({current_task.name.upper()}) MASTERED AND GENERALIZED!")

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
                print(f"💾 Checkpoint saved: {ckpt_path}")

                replay_buffer.archive_task(level=old_level, train_data=raw_train_data, val_data=raw_val_data)

                # Reset optimizer weight decay according to level_configs
                next_lvl_cfg = curriculum_cfg["curriculum"].get("level_configs", {}).get(teacher.current_level, {})
                next_task_wd = float(next_lvl_cfg.get("base_weight_decay", 1.0 if teacher.current_level in [1, 2] else 0.01))
                for g in optimizer.param_groups:
                    g["weight_decay"] = next_task_wd
                teacher.base_weight_decay = next_task_wd

                current_task = get_task_for_level(teacher.current_level)
                raw_train_data, raw_val_data = current_task.get_data()

                active_train_data = replay_buffer.mix_dataset(raw_train_data, max_samples=4000)
                is_modular = "modular" in current_task.name
                is_modular_mult = "modular_mult" in current_task.name
                trainer.set_datasets(active_train_data, raw_val_data, is_modular_task=is_modular, force_full_batch=is_modular_mult)

                val_em_smoother = RollingMetric(window_size=5)

                print(f"⏩ PROMOTING TO NEXT LEVEL {teacher.current_level}/{max_levels}: {current_task.name.upper()}")
                print(f"   Training pool: {len(active_train_data)} examples | Val pool: {len(raw_val_data)} examples")
                print(f"   Execution mode: {'FULL-BATCH' if trainer.active_full_batch else f'MINI-BATCH (size={trainer.batch_size})'}")
                print("=" * 85 + "\n")

            elif meta.get("status") == "curriculum_mastered":
                total_time = (time.time() - start_time) / 60
                print("\n" + "*" * 85)
                print(f"🎓 CURRICULUM MASTERED (ALL {max_levels} LEVELS)!")
                print(f"TIME: {total_time:.2f} MIN | STEP: {step}")
                print(f"Final checkpoint saved: checkpoints/final_model_mastered.pt")
                print("*" * 85)

                torch.save(model.state_dict(), "checkpoints/final_model_mastered.pt")
                break


if __name__ == "__main__":
    main()