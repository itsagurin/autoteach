import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import yaml

from src.models.tokenizer import MathTokenizer
from src.models.nanogpt import MicroNanoGPT
from src.engine.masking import prepare_batch
from src.tasks.modular import ModularArithmeticTask
from src.telemetry.metrics import compute_weight_norm, compute_exact_match, RollingMetric
from src.telemetry.logger import JSONLinesLogger


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device in use: {device}")

    with open("configs/student_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    tokenizer = MathTokenizer()
    model = MicroNanoGPT(
        vocab_size=tokenizer.vocab_size,
        d_model=config["model"]["d_model"],
        n_layer=config["model"]["n_layer"],
        n_head=config["model"]["n_head"],
        block_size=config["model"]["block_size"]
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["lr"]),
        betas=tuple(config["training"]["betas"]),
        weight_decay=float(config["training"]["weight_decay"])
    )

    task = ModularArithmeticTask(p=97, op="*", train_ratio=0.25, seed=42)
    train_data, val_data = task.get_data()
    print(f"Examples in Train: {len(train_data)} | In Val: {len(val_data)}")

    x_train, y_train = prepare_batch(train_data, tokenizer, device)
    x_val, y_val = prepare_batch(val_data, tokenizer, device)

    logger = JSONLinesLogger(filepath="logs/grok_run.jsonl")
    val_em_smoother = RollingMetric(window_size=5)

    print("\nStarting run for modular multiplication (p=97)...")
    print(f"{'Step':<7} | {'Train Loss':<10} | {'Train EM':<10} | {'Val EM':<10} | {'Smooth Val':<10} | {'||W||_2':<7}")
    print("-" * 68)

    max_steps = 40000
    eval_interval = 200

    for step in range(max_steps + 1):
        model.train()
        optimizer.zero_grad()

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

            logger.log(step, {
                "train_loss": round(loss.item(), 5),
                "val_loss": round(val_loss.item(), 5),
                "train_em": round(train_em, 4),
                "val_em": round(val_em, 4),
                "smooth_val_em": round(smooth_val_em, 4),
                "weight_norm": round(w_norm, 2)
            })

            print(f"{step:<7} | {loss.item():<10.4f} | {train_em:<10.2%} | {val_em:<10.2%} | {smooth_val_em:<10.2%} | {w_norm:<7.2f}")

            if smooth_val_em >= 0.98 and step > 1000:
                print(f"\n[CONFIRMED GROKKING] Step {step}: Generalization reached 98%+ EM!")
                break


if __name__ == "__main__":
    main()